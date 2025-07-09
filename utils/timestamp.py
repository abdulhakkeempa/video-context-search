def convert_second_to_timestamp(seconds: int) -> str:
    mm = seconds // 60
    ss = seconds % 60
    hh = mm // 60
    mm = mm % 60

    return f"{hh:02}:{mm:02}:{ss:02}" if hh > 0 else f"{mm:02}:{ss:02}"