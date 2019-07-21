################################
# Imports
################################

import xml.etree.ElementTree as ET 


################################
# Parse XML
################################
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    flat_xml = {elem.tag: elem.text.strip() for elem in root.iter()}
    return flat_xml


################################
# Time string
################################

def time2str(t):
    str_time = ''
    t_day = int(t // (60 * 60 * 24))
    t_hour = int((t - t_day * 60 * 60 * 24) // (60 * 60))
    t_min = int((t - t_day * 60 * 60 * 24 - t_hour * 60 * 60) // 60)
    t_sec = (t - t_day * 60 * 60 * 24 - t_hour * 60 * 60 - t_min * 60)
    
    # Format time
    if t_day > 0:
        str_time += '{} d'.format(t_day)
        str_time += ' - {} h'.format(t_hour)
        str_time += ' - {} m'.format(t_min)
        str_time += ' - {} s'.format(t_sec)
    if (t_day == 0) and (t_hour > 0):
        str_time += '{} h'.format(t_hour)
        str_time += ' - {} m'.format(t_min)
        str_time += ' - {} s'.format(t_sec)
    if (t_day == 0) and (t_hour == 0) and (t_min > 0):
        str_time += '{} m'.format(t_min)
        str_time += ' - {} s'.format(t_sec)
    if (t_day == 0) and (t_hour == 0) and (t_min == 0) and (t_sec > 0):
        str_time += '{} s'.format(t_sec)
    return str_time