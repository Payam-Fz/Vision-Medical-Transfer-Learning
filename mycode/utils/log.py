from datetime import datetime

def print_time(t):
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        t (int): time period in seconds
    Returns:
        s (string): time period formatted
    """
    h = t//3600
    m = (t%3600)//60
    s = (t%3600)%60
    return '%dh:%dm:%ds'%(h,m,s)

def get_curr_datetime():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H%M")
    return dt_string

def print_log(*args, sep=' '):
    text = sep.join(map(str, args))
    lines = text.split('\n')
    for line in lines:
        print('@_@\t' + line)