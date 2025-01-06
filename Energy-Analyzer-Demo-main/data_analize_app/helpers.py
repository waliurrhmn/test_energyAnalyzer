
def time_to_minutes(hour):
    """
        convert hour to minute time 
    """
    return hour.hour * 60 + hour.minute


def validate_time_periods(peak_start, peak_end, offpeak_start, offpeak_end):
    peak_start_mins = time_to_minutes(peak_start)
    peak_end_mins = time_to_minutes(peak_end)
    offpeak_start_mins = time_to_minutes(offpeak_start)
    offpeak_end_mins = time_to_minutes(offpeak_end)

    # Check if start times are the same
    if peak_start_mins == offpeak_start_mins:
        raise ValueError("Peak and off-peak start times cannot be the same")
    return True