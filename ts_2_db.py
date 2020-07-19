# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:24:30 2020

@author: solis

Exporta los ficheros de salidas de idw generados en una db sqlite

file_point: fichero de puntos interpolados
orgs: es un diccionario cuya key es el nombre de la tabla a importar
    y el valor es una tupla cuyo primer elemento es el fichero de puntos
    donde se ha realizado la interpolación de la variable
    y cuyo segundo elemento el el fichero con los datos interpolados
dst: directorio del fichero de salida -no el nombre-
separator: de las columnas de los ficheros csv a exportar
"""
import littleLogging as logging

dir_org = r'H:\off\phdhs2127\recarga\cc_q\meteo_out'
orgs = {
        'pd': ('pd_interpolated_points.csv', 'pd_idw.csv'),
        'tmax': ('tmaxd_interpolated_points.csv', 'tmaxd_idw.csv'),
        'tmin': ('tmaxd_interpolated_points.csv', 'tmind_idw.csv')
        }
dst = r'H:\off\phdhs2127\recarga\cc_q\meteo_out'
separator = ';'

if __name__ == "__main__":

    try:
        from datetime import datetime
        from time import time
        import traceback
        from interpol import output_files_2db

        now = datetime.now()

        startTime = time()

        output_files_2db(dir_org, orgs, dst)

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
