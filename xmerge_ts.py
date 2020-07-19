# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:24:30 2020

@author: solis

Permite combinar diversos ficheros de salidas de idw generados desde main
    en uno solo; tiene interés si todos los ficheros tienen los mismos
    puntos de interpolación con el mismo rango de fechas
"""
import littleLogging as logging

orgs = (r'H:\off\phdhs2127\recarga\cc_q\meteo_out\point_pd_idw.csv',
        r'H:\off\phdhs2127\recarga\cc_q\meteo_out\point_tmaxd_idw.csv',
        r'H:\off\phdhs2127\recarga\cc_q\meteo_out\point_tmind_idw.csv')
header = ('point', 'fecha', 'pd', 'tdmin', 'tdmax')
dst = r'H:\off\phdhs2127\recarga\cc_q\meteo_out\p_tmax_tmin_idw.csv'
separator = ';'

if __name__ == "__main__":

    try:
        from datetime import datetime
        from time import time
        import traceback
        from interpol import merge_idw_output_files as merge

        now = datetime.now()

        startTime = time()

        merge(orgs, header, dst, separator)

        xtime = time() - startTime
        print(f'El script tardó {xtime:0.0f} s')

    except ValueError:
        msg = traceback.format_exc()
        logging.append(f'ValueError exception\n{msg}')
    except ImportError:
        msg = traceback.format_exc()
        print (f'ImportError exception\n{msg}')
    except Exception:
        msg = traceback.format_exc()
        logging.append(f'Exception\n{msg}')
    finally:
        logging.dump()
        print('\nFin')

