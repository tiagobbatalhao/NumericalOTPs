import numpy as py
import ImplementationOTPs as implement_module

def load_data(n_copies):
    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    interested = {
        '00': '00xx',
        '01': '01xx',
        '10': '10xx',
        '11': '11xx',
        '0x': '0xxx',
        '1x': '1xxx',
        'x0': 'x0xx',
        'x1': 'x1xx',
        'xx': 'xxxx',
    }
    rhos = {}
    for new_label, old_label in interested.items():
        filename = template.format(old_label, n_copies)
        rhos[new_label] = implement_module.load_from_file(filename)
    rhos['PARITY_E'] = (rhos['00'] + rhos['11']) * 0.5
    rhos['PARITY_O'] = (rhos['01'] + rhos['10']) * 0.5
    return rhos
if 'data' not in locals():
    data = load_data(7)

def green_line():
    rho = ( data['0x'] + (-1 * data['1x']) )
    filename = 'Data/ForPaper_GreenLine.txt'
    implement_module.write_to_file(rho, filename)
    return rho

def lower_red_line():
    rho = ( data['PARITY_E'] + (-1 * data['PARITY_O']) )
    filename = 'Data/ForPaper_LowerRedLine.txt'
    implement_module.write_to_file(rho, filename)
    return rho

if 'rho_green' not in locals():
    rho_green = green_line()
if 'rho_lowerred' not in locals():
    rho_lowerred = lower_red_line()
