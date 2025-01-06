# Energy Analyzer Documentation

## Overview

This documentation covers the functionality of the Energy Analyzer application.

---

## [Frontend Functions (`upload.html`)](data_analize_app/templates/upload.html)

### **Data Filtering and Processing Functions**

- **`filterDataByTime(data, filterType)`**
  - **Purpose**: Filters energy consumption data based on different time periods.
  - **Parameters**:
    - `data`: Raw energy consumption data.
    - `filterType`: Type of time filter (daily, weekly, monthly, quarterly).
  - **Usage**: Called when changing time period filters in the UI.

- **`updateConsumptionAverages(averages)`**
  - **Purpose**: Updates the display of consumption averages in the UI.
  - **Parameters**:
    - `averages`: Object containing different time-based consumption averages.
  - **Usage**: Called after data processing to show average consumption patterns.

- **`updateTimeFilter()`**
  - **Purpose**: Handles time filter changes and updates chart display.
  - **Usage**: Connected to time filter dropdown change events.

### **Chart Management Functions**

- **`createDatasets(data, chartType)`**
  - **Purpose**: Creates chart datasets based on consumption data.
  - **Parameters**:
    - `data`: Processed energy data.
    - `chartType`: Type of chart to display (line, bar, etc.).
  - **Usage**: Called when rendering or updating charts.

- **`updateChartType()`**
  - **Purpose**: Handles switching between different chart visualizations.
  - **Usage**: Called when user changes chart type in UI.

- **`createChartConfig(chartType, labels, datasets, indices)`**
  - **Purpose**: Generates Chart.js configuration.
  - **Parameters**:
    - `chartType`: Type of chart.
    - `labels`: X-axis labels.
    - `datasets`: Chart data.
    - `indices`: Data indices for annotations.
  - **Usage**: Used internally by chart rendering functions.

### **Utility Functions**

- **`formatEUNumber(number, decimals = 2)`**
  - **Purpose**: Formats numbers in European format.
  - **Parameters**:
    - `number`: Number to format.
    - `decimals`: Decimal places (default: 2).
  - **Usage**: Used for displaying numerical values in the UI.

- **`calculateAdjustedCeiling()`**
  - **Purpose**: Calculates adjusted power consumption ceiling.
  - **Usage**: Used for setting consumption thresholds in charts.

---

## [Backend Functions (`views.py`)](data_analize_app/views.py)

### **Core Data Processing**

- **`analyze_energy_data(df, pricing_periods, kwh_ceiling)`**
  - **Purpose**: Main function for analyzing energy consumption data.
  - **Parameters**:
    - `df`: Pandas DataFrame containing energy data.
    - `pricing_periods`: Dictionary defining peak/off-peak hours.
    - `kwh_ceiling`: Maximum power consumption threshold.
  - **Returns**: Dictionary containing analyzed data and statistics.
  - **Usage**: Called after file upload to process energy data.

- **`clean_dataframe(df)`**
  - **Purpose**: Cleans and validates input data.
  - **Parameters**:
    - `df`: Raw DataFrame from uploaded file.
  - **Usage**: Called before data analysis to ensure data quality.

### **File Handling**

- **`read_file(file)`**
  - **Purpose**: Reads and validates uploaded energy data files.
  - **Parameters**:
    - `file`: Uploaded file object.
  - **Usage**: Called during file upload process.

- **`upload_file(request)`**
  - **Purpose**: Handles file upload requests.
  - **Parameters**:
    - `request`: HTTP request object.
  - **Usage**: Main entry point for file uploads.

### **Data Analysis Functions**

- **`calculate_data_summary(df, file_name, pricing_periods)`**
  - **Purpose**: Generates summary statistics for uploaded data.
  - **Parameters**:
    - `df`: Processed DataFrame.
    - `file_name`: Name of uploaded file.
    - `pricing_periods`: Time period configurations.
  - **Usage**: Called after data processing to generate summary statistics.

- **`calculate_consumption_averages(df)`**
  - **Purpose**: Calculates consumption averages for different time periods.
  - **Parameters**:
    - `df`: Processed DataFrame.
  - **Usage**: Called to generate average consumption statistics.

---

## Adding New Features

### **Adding New Chart Filters**

1. Add new filter option in `upload.html`:

   ```html
   <select class="form-control" id="newFilter">
     <option value="option1">Option 1</option>
     ...
   </select>
    ```

2. Create new filter function in JavaScript:

    ```javascript
    function handleNewFilter(data) {
        // Filter implementation
    }
    ```

3. Update `updateChartType()` to include the new filter.

**Adding New Chart Types**

1. Add new chart type option in HTML.

2. Extend `createDatasets()` to handle the new chart type.

3. Add configuration in `createChartConfig()`.

4. Update chart rendering logic in `renderChart()`.

**Adding New Analysis Features**  

1. Implement new analysis function in `views.py`.

2. Add corresponding frontend function in `upload.html`.

3. Update data processing pipeline in `analyze_energy_data()`.

4. Add new visualization components if needed.

---

## Best Practices

- Always validate input data before processing.

- Use appropriate error handling for file operations.

- Implement proper data cleaning before analysis.

- Follow existing code structure for consistency.

- Document new features and functions.

- Test new features with various data scenarios.

---

This documentation provides a comprehensive overview of the codebase and instructions for extending functionality.
