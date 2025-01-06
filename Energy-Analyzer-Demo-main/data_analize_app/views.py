# views.py
import time
import pandas as pd
from datetime import datetime

from django.http import JsonResponse
from django.contrib import messages
from django.shortcuts import render,redirect

from .models import EnergyData, UserSettings
from .helpers import validate_time_periods


def home(request):
    return render(request,'home.html')

def analyze_energy_data(df, pricing_periods, kwh_ceiling):
    """Analyze energy data with custom peak/off-peak time periods"""
    start_time = time.time()
    initial_rows = len(df)
    
    # Convert to datetime and set as index
    df['Datum tijd'] = pd.to_datetime(df['Datum tijd'])
    
    # Debug prints
    print("\n=== Debug Information ===")
    print(f"Initial data range:")
    print(f"Start: {df['Datum tijd'].min()}")
    print(f"End: {df['Datum tijd'].max()}")
    print(f"\nSelected period:")
    print(f"Start: {pricing_periods['high_price']['start']} (type: {type(pricing_periods['high_price']['start'])})")
    print(f"End: {pricing_periods['high_price']['end']} (type: {type(pricing_periods['high_price']['end'])})")
    
    # Get the data's date range
    data_start = df['Datum tijd'].min()
    data_end = df['Datum tijd'].max()
    
    # Check if selected date range overlaps with data range
    if (pricing_periods['high_price']['end'] < data_start or 
        pricing_periods['high_price']['start'] > data_end):
        raise ValueError(
            f"Selected period ({pricing_periods['high_price']['start']} to "
            f"{pricing_periods['high_price']['end']}) is outside the available data range "
            f"({data_start} to {data_end}). Please select a period within the available range."
        )

    print(f"\nBefore filtering: {len(df)} rows")

    # Filter data within the selected date range
    filtered_df = df[(df['Datum tijd'] >= pricing_periods['high_price']['start']) & 
                    (df['Datum tijd'] <= pricing_periods['high_price']['end'])]
    print(f"After filtering: {len(filtered_df)} rows")
    print(f"Filtered data range:")
    if not filtered_df.empty:
        print(f"Start: {filtered_df['Datum tijd'].min()}")
        print(f"End: {filtered_df['Datum tijd'].max()}")
    print("=== End Debug Information ===\n")
    
    df = filtered_df
    
    # Check if any data remains after filtering
    if len(df) == 0:
        raise ValueError(
            f"No data found for the selected date range. Available data range is from "
            f"{filtered_df['Datum tijd'].min()} to {filtered_df['Datum tijd'].max()}."
        )
    
    # Check for duplicates before processing
    duplicate_mask = df['Datum tijd'].duplicated(keep=False)
    duplicate_timestamps = int(duplicate_mask.sum())
    
    # Get all duplicate timestamps
    duplicate_times = []
    if duplicate_timestamps > 0:
        duplicate_df = df[duplicate_mask].copy()
        # Group by timestamp and get counts
        dup_counts = duplicate_df['Datum tijd'].value_counts()
        # Format all duplicates with their counts
        duplicate_times = [
            {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'count': int(count)
            }
            for ts, count in dup_counts.items()
        ]
        # Sort by timestamp
        duplicate_times.sort(key=lambda x: x['timestamp'])
    
    # Extract hour and minute for time comparison
    df['hour'] = df['Datum tijd'].dt.hour
    df['minute'] = df['Datum tijd'].dt.minute
    df['time_value'] = df['hour'] * 60 + df['minute']  # Convert to minutes for easier comparison
    
    # Convert peak and off-peak times to minutes for comparison
    peak_start = pricing_periods['peak_hours']['start']  # Format: "HH:MM"
    peak_end = pricing_periods['peak_hours']['end']
    offpeak_start = pricing_periods['offpeak_hours']['start']
    offpeak_end = pricing_periods['offpeak_hours']['end']
    
    def time_to_minutes(time_str):
        hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute
    
    peak_start_minutes = time_to_minutes(peak_start)
    peak_end_minutes = time_to_minutes(peak_end)
    offpeak_start_minutes = time_to_minutes(offpeak_start)
    offpeak_end_minutes = time_to_minutes(offpeak_end)
    
    # Create peak/off-peak masks
    if peak_start_minutes < peak_end_minutes:
        # Normal case (e.g., 08:00 to 20:00)
        peak_mask = (df['time_value'] >= peak_start_minutes) & (df['time_value'] < peak_end_minutes)
    else:
        # Overnight case
        peak_mask = (df['time_value'] >= peak_start_minutes) | (df['time_value'] < peak_end_minutes)
    
    if offpeak_start_minutes < offpeak_end_minutes:
        # Normal case
        offpeak_mask = (df['time_value'] >= offpeak_start_minutes) & (df['time_value'] < offpeak_end_minutes)
    else:
        # Overnight case (e.g., 20:00 to 08:00)
        offpeak_mask = (df['time_value'] >= offpeak_start_minutes) | (df['time_value'] < offpeak_end_minutes)
    
    # Create separate peak and off-peak consumption columns
    df['Day_Consumption'] = 0.0  # Peak hours consumption
    df['Night_Consumption'] = 0.0  # Off-peak hours consumption
    
    # Set values based on masks
    df.loc[peak_mask, 'Day_Consumption'] = df.loc[peak_mask, 'Aplus (consumptie)']
    df.loc[offpeak_mask, 'Night_Consumption'] = df.loc[offpeak_mask, 'Aplus (consumptie)']
    
    # Set index and handle duplicates by summing consumption
    df.set_index('Datum tijd', inplace=True)
    df = df.groupby(df.index).agg({
        'Day_Consumption': 'sum',
        'Night_Consumption': 'sum'
    })
    
    # Calculate costs
    price_high = pricing_periods['prices']['high']
    price_low = pricing_periods['prices']['low']
    
    df['Day_Cost'] = df['Day_Consumption'] * price_high
    df['Night_Cost'] = df['Night_Consumption'] * price_low
    
    filtered_rows = len(df)
    
    # Resample to daily data for the chart
    daily = df.resample('D').agg({
        'Day_Consumption': 'sum',
        'Night_Consumption': 'sum',
        'Day_Cost': 'sum',
        'Night_Cost': 'sum'
    })
    
    # Calculate totals
    total_day_consumption = daily['Day_Consumption'].sum()
    total_night_consumption = daily['Night_Consumption'].sum()
    total_day_cost = daily['Day_Cost'].sum()
    total_night_cost = daily['Night_Cost'].sum()
    
    # Calculate processing time
    processing_time = round(time.time() - start_time, 2)
    
    # Generate hourly data
    df_hourly = df.copy()
    df_hourly.index = df_hourly.index.floor('h')
    hourly_data = df_hourly.groupby(df_hourly.index).agg({'Day_Consumption': 'sum', 'Night_Consumption': 'sum'})
    hourly_labels = hourly_data.index.strftime('%Y-%m-%d %H:%M').tolist()
    hourly_day_values = hourly_data['Day_Consumption'].tolist()
    hourly_night_values = hourly_data['Night_Consumption'].tolist()
    
    # Prepare chart data
    chart_data = {
        'labels': daily.index.strftime('%Y-%m-%d').tolist(),
        'datasets': {
            'consumption': {
                'day_consumption': [float(x) for x in daily['Day_Consumption'].tolist()],
                'night_consumption': [float(x) for x in daily['Night_Consumption'].tolist()]
            },
            'costs': {
                'day_costs': [float(x) for x in daily['Day_Cost'].tolist()],
                'night_costs': [float(x) for x in daily['Night_Cost'].tolist()]
            },            
            'hourly_data':{
              'labels': hourly_labels,
              'day_values': [float(x) for x in hourly_day_values],
              'night_values': [float(x) for x in hourly_night_values],
            }
        },
        'summary': {
            'total_day_consumption': float(total_day_consumption),
            'total_night_consumption': float(total_night_consumption),
            'total_day_cost': float(total_day_cost),
            'total_night_cost': float(total_night_cost),
            'total_consumption': float(total_day_consumption + total_night_consumption),
            'total_cost': float(total_day_cost + total_night_cost),
            'peak_price': float(price_high),
            'off_peak_price': float(price_low)

        },
        'statistics': {
            'total_records': int(initial_rows),
            'processed_records': int(filtered_rows),
            'duplicate_timestamps': int(duplicate_timestamps),
            'duplicate_examples': duplicate_times,
            'processing_time_seconds': float(processing_time),
            'date_range': {
                'start': pricing_periods['high_price']['start'].strftime('%Y-%m-%d %H:%M'),
                'end': pricing_periods['high_price']['end'].strftime('%Y-%m-%d %H:%M')
            }
        },
        'ceiling': float(kwh_ceiling)
    }
    
    return chart_data

def clean_dataframe(df):
    """Clean the dataframe by removing summary rows and validating data"""
    print("Initial DataFrame shape:", df.shape)
    print("Initial columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    
    # Remove rows where 'Datum tijd' contains 'Totalen'
    df = df[~df['Datum tijd'].astype(str).str.contains('Totalen', case=False, na=False)]
    
    print("\nShape after removing 'Totalen' rows:", df.shape)
    
    # Convert consumption to numeric, handling both string and numeric values
    if df['Aplus (consumptie)'].dtype == 'object':
        # If the column contains strings (with possible commas)
        df['Aplus (consumptie)'] = df['Aplus (consumptie)'].astype(str).str.replace(',', '').astype(float)
    else:
        # If the column is already numeric
        df['Aplus (consumptie)'] = df['Aplus (consumptie)'].astype(float)
    
    print("\nShape after cleaning consumption data:", df.shape)
    print("\nFinal cleaned data sample:")
    print(df.head())
    
    return df

def read_file(file):
    """Read and validate the uploaded file"""
    try:
        # Determine file type from extension
        file_extension = file.name.lower().split('.')[-1]
        
        # Read file based on extension
        if file_extension == 'csv':
            for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file, encoding=encoding, usecols=['Datum tijd', 'Aplus (consumptie)'])
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
                except pd.errors.ParserError:
                    raise ValueError(
                        "Missing required columns. Please ensure your file has "
                        "'Datum tijd' and 'Aplus (consumptie)' columns."
                    )
            if 'df' not in locals():
                raise ValueError("Unable to read the CSV file with supported encodings")
        elif file_extension == 'xlsx':
            try:
                df = pd.read_excel(file, usecols=['Datum tijd', 'Aplus (consumptie)'])
            except ValueError:
                raise ValueError(
                    "Missing required columns. Please ensure your file has "
                    "'Datum tijd' and 'Aplus (consumptie)' columns."
                )
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please upload a CSV or XLSX file.")

        # Print initial data for debugging
        print("\nInitial data types:")
        print(df.dtypes)
        print("\nFirst few dates before processing:")
        print(df['Datum tijd'].head())

        # Remove summary rows efficiently
        df = df[~df['Datum tijd'].astype(str).str.contains('Totalen|Total|Sum', case=False, na=False)]
        
        # Convert consumption to numeric efficiently
        try:
            df['Aplus (consumptie)'] = pd.to_numeric(
                df['Aplus (consumptie)'].astype(str).str.replace(',', ''),
                errors='coerce'
            )
            if df['Aplus (consumptie)'].isna().any():
                print("\nInvalid numeric values found in these rows:")
                print(df[df['Aplus (consumptie)'].isna()])
                raise ValueError("Invalid numeric values found in 'Aplus (consumptie)' column")
        except Exception as e:
            print(f"\nError converting consumption values: {str(e)}")
            raise ValueError(
                "Invalid numeric values in 'Aplus (consumptie)' column. "
                "Please ensure all values are numbers."
            )

        # Efficient date parsing with detailed error reporting
        try:
            print("\nAttempting to parse dates...")
            
            # Clean the date strings first
            df['Datum tijd'] = df['Datum tijd'].astype(str).str.strip()
            
            print("\nSample of date strings after cleaning:")
            print(df['Datum tijd'].head())
            
            # Try parsing with the actual format from the data (YYYY-MM-DD HH:MM:SS)
            df['Datum tijd'] = pd.to_datetime(df['Datum tijd'], errors='coerce')
            
            # Check for any invalid dates
            nat_mask = df['Datum tijd'].isna()
            if nat_mask.any():
                print("\nRows with invalid dates:")
                print(df[nat_mask]['Datum tijd'])
                raise ValueError(
                    "Invalid date format in 'Datum tijd' column. "
                    "Please ensure dates are in a valid format (e.g., YYYY-MM-DD HH:MM:SS or M/D/YYYY H:MM)"
                )
                
            print("\nSuccessfully parsed dates. Sample of parsed dates:")
            print(df['Datum tijd'].head())
            
        except Exception as e:
            print(f"\nError parsing dates: {str(e)}")
            raise ValueError(
                "Invalid date format in 'Datum tijd' column. "
                "Please ensure dates are in a valid format (e.g., YYYY-MM-DD HH:MM:SS or M/D/YYYY H:MM)"
            )
        
        return df
        
    except Exception as e:
        # Re-raise with more context if needed
        if isinstance(e, ValueError):
            raise
        print(f"\nUnexpected error: {str(e)}")
        raise ValueError(
            f"Error processing file: {str(e)}. "
            "Please ensure your file follows the required format."
        )

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        try:
            print("\n=== Starting file upload process ===")
            print(f"File name: {request.FILES['file'].name}")
            print(f"File size: {request.FILES['file'].size} bytes")
            
            # Get user settings from form
            peak_hours_start = request.POST.get('peak_hours_start')
            offpeak_hours_start = request.POST.get('offpeak_hours_start')

            # Validate Pricing & Time Ranges
            try:
                # Convert time strings to datetime.time objects
                peak_start = datetime.strptime(peak_hours_start, '%H:%M').time()
                offpeak_start = datetime.strptime(offpeak_hours_start, '%H:%M').time()
                # Set end times based on the other period's start time
                peak_end = offpeak_start
                offpeak_end = peak_start

                # Validate the time periods
                validate_time_periods(peak_start, peak_end, offpeak_start, offpeak_end)

                # Get the ceiling value
                kwh_ceiling = float(request.POST.get('kwh_ceiling', 10))
                if kwh_ceiling <= 0:
                    raise ValueError("Quarter-hourly Power Ceiling must be greater than 0.")

                # Validate prices
                price_high = float(request.POST.get('price_high', 0))
                price_low = float(request.POST.get('price_low', 0))
                if price_high < 0 or price_low < 0:
                    raise ValueError("Prices cannot be negative.")

                print("Time ranges validation successful")
                print(f"Prices validated - High: {price_high}, Low: {price_low}")
            except ValueError as e:
                print(f"Pricing & Time Ranges validation error: {str(e)}")
                if "cannot be negative" in str(e):
                    raise ValueError(f"Price Error: {str(e)}")
                else:
                    raise ValueError(f"Time Ranges Error: {str(e)}")

            # Read the file first to get the date range
            df = read_file(request.FILES['file'])
            
            # Use the full date range from the file
            high_price_start = df['Datum tijd'].min()
            high_price_end = df['Datum tijd'].max()
            
            pricing_periods = {
                'high_price': {
                    'start': high_price_start,
                    'end': high_price_end,
                    'start_time': high_price_start.time(),
                    'end_time': high_price_end.time()
                },
                'peak_hours': {
                    'start': peak_hours_start,
                    'end': offpeak_hours_start
                },
                'offpeak_hours': {
                    'start': offpeak_hours_start,
                    'end': peak_hours_start
                },
                'prices': {
                    'high': price_high,
                    'low': price_low
                }
            }
            
            print("\n=== Starting file processing ===")
            # Remove the nested try-except and combine with outer one
            chart_data = analyze_energy_data(df, pricing_periods, kwh_ceiling)
            print("\nData analysis successful")
            
            # Calculate data summary
            data_summary = calculate_data_summary(df, request.FILES['file'].name, pricing_periods)

            averages = calculate_consumption_averages(df)

            # Prepare response with both chart data and summary
            response_data = {
                'success': True,
                'chart_data': chart_data,
                'kwh_ceiling': kwh_ceiling,
                'consumption_averages': averages,
            }
            response_data.update(data_summary)
            
            return JsonResponse(response_data)

        except ValueError as e:
            print(f"\nValidation error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return JsonResponse({
                'success': False,
                'error': f"An unexpected error occurred: {str(e)}"
            })
    
    return render(request, 'upload.html')

def calculate_data_summary(df, file_name, pricing_periods):
    try:
        # Calculate date range
        start_date = df['Datum tijd'].min().strftime('%Y-%m-%d %H:%M')
        end_date = df['Datum tijd'].max().strftime('%Y-%m-%d %H:%M')
        
        # Calculate basic statistics
        total_records = len(df)
        missing_values = df['Aplus (consumptie)'].isna().sum()
        duplicate_mask = df['Datum tijd'].duplicated(keep=False)
        duplicate_records = int(duplicate_mask.sum())
        
        # Calculate consumption statistics
        daily_consumption = df.groupby(df['Datum tijd'].dt.date)['Aplus (consumptie)'].sum()
        avg_daily_consumption = daily_consumption.mean()
        max_consumption = df['Aplus (consumptie)'].max()
        min_consumption = df['Aplus (consumptie)'].min()
        
        # Calculate peak vs off-peak distribution
        df['hour'] = df['Datum tijd'].dt.hour
        
        # Get the peak/off-peak hours from the pricing_periods
        peak_start = int(pricing_periods['peak_hours']['start'].split(':')[0])
        peak_end = int(pricing_periods['peak_hours']['end'].split(':')[0])
        
        # Determine peak hours
        if peak_start < peak_end:
            peak_mask = df['hour'].between(peak_start, peak_end - 1)
        else:
            # For overnight peak hours (e.g., 18:00 to 00:00)
            offpeak_start = int(pricing_periods['offpeak_hours']['start'].split(':')[0])
            offpeak_end = int(pricing_periods['offpeak_hours']['end'].split(':')[0])
            peak_mask = ~df['hour'].between(offpeak_start, offpeak_end - 1)
        
        # Calculate consumption for peak and off-peak
        peak_consumption = df[peak_mask]['Aplus (consumptie)'].sum()
        total_consumption = df['Aplus (consumptie)'].sum()
        
        # Calculate percentages
        peak_percentage = peak_consumption / total_consumption if total_consumption > 0 else 0
        off_peak_percentage = 1 - peak_percentage
        
        # Convert values to appropriate scale if needed
        scale_factor = 1  # Adjust this if needed (e.g., 1000 for Wh to kWh)
        
        # Debug prints
        print("\n=== Data Summary Debug ===")
        print(f"Total Records: {total_records}")
        print(f"Daily Consumption Stats:")
        print(f"  Average: {avg_daily_consumption / scale_factor:.2f} kWh")
        print(f"  Max: {max_consumption / scale_factor:.2f} kWh")
        print(f"  Min: {min_consumption / scale_factor:.2f} kWh")
        print(f"Peak Hours: {peak_percentage * 100:.1f}%")
        print(f"Off-Peak Hours: {off_peak_percentage * 100:.1f}%")
        print("========================")
        
        return {
            'file_name': file_name,
            'total_records': total_records,
            'start_date': start_date,
            'end_date': end_date,
            'missing_values': int(missing_values),
            'duplicate_records': duplicate_records,
            'avg_daily_consumption': float(avg_daily_consumption / scale_factor),
            'max_consumption': float(max_consumption / scale_factor),
            'min_consumption': float(min_consumption / scale_factor),
            'peak_hours_percentage': float(peak_percentage),
            'off_peak_hours_percentage': float(off_peak_percentage)
        }
    except Exception as e:
        print(f"Error in calculate_data_summary: {str(e)}")
        return {
            'file_name': file_name,
            'total_records': len(df),
            'start_date': '-',
            'end_date': '-',
            'missing_values': 0,
            'duplicate_records': 0,
            'avg_daily_consumption': 0,
            'max_consumption': 0,
            'min_consumption': 0,
            'peak_hours_percentage': 0,
            'off_peak_hours_percentage': 0
        }

def calculate_consumption_averages(df):
    """
    Calculate hourly, daily, weekly, and monthly averages from 15-minute interval data
    
    Args:
        df: DataFrame with 'Datum tijd' and 'Aplus (consumptie)' columns
    
    Returns:
        dict: Dictionary containing all averages
    """
    # Convert 'Datum tijd' to datetime if it's not already
    df['Datum tijd'] = pd.to_datetime(df['Datum tijd'])
    
    # Calculate hourly averages (0-23)
    hourly_avg = df.groupby(df['Datum tijd'].dt.hour)['Aplus (consumptie)'].mean()
    
    # Calculate daily averages (Monday-Sunday)
    daily_avg_by_day = df.groupby(df['Datum tijd'].dt.day_name())['Aplus (consumptie)'].mean()
    daily_avg = daily_avg_by_day.mean()
    
    # TOGGLE COMMENT ON NEXT TWO LINES TO INCLUDE/EXCLUDE 2022 DATA
    SHOW_2023_ONLY = False  # Set to False to include 2022 data
    df_filtered = df[df['Datum tijd'].dt.year == 2023] if SHOW_2023_ONLY else df
    
    # Calculate weekly averages with ISO calendar
    isocalendar = df_filtered['Datum tijd'].dt.isocalendar()
    weekly_avg_by_week = df_filtered.groupby([isocalendar.year, isocalendar.week])['Aplus (consumptie)'].mean()
    
    # Additional filter to ensure we only get the desired year's data
    if SHOW_2023_ONLY:
        weekly_avg_by_week = weekly_avg_by_week[weekly_avg_by_week.index.get_level_values(0) == 2023]
    
    weekly_avg = weekly_avg_by_week.mean()
    
    # Create weekly averages dictionary with proper formatting
    weekly_avg_dict = {f"Week {week:02d}, {year}": value 
                      for (year, week), value in weekly_avg_by_week.items()}
    
    # Calculate monthly averages (January-December)
    monthly_avg_by_month = df.groupby(df['Datum tijd'].dt.month_name())['Aplus (consumptie)'].mean()
    monthly_avg = monthly_avg_by_month.mean()
    
    return {
        'hourly_avg': hourly_avg.round(2).to_dict(),
        'daily_avg': daily_avg.round(2),
        'daily_avg_by_day': daily_avg_by_day.round(2).to_dict(),
        'weekly_avg': weekly_avg.round(2),
        'weekly_avg_by_week': weekly_avg_dict,
        'monthly_avg': monthly_avg.round(2),
        'monthly_avg_by_month': monthly_avg_by_month.round(2).to_dict()
    }

def login_view(request):
    if request.method == 'POST':
        password = request.POST.get('password')
        if password == "energy123":  # same password as in middleware
            request.session['is_authenticated'] = True
            return redirect('home')
        else:
            messages.error(request, 'Invalid password')
    return render(request, 'login.html')