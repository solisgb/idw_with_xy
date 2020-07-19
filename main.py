# -*- coding: utf-8 -*-
"""
Created on 06/09/2019

@author: Luis Solís

driver módulo interpol
Interpolación de valores de puntos en un rango de fechas
Antes de ejecutar la app asegúrate de que rellenas correctamente los valores
    de los parámetros en el fichero xml
    Puedes ejecutar el programa con un fichero xml
    de cualquier nombre, pero el contenido es fijo
Al finalizar la ejecución el programa crea el fichero app.log con las
    incidencias de la ejecución
Antes de ejecutar la app rellena los contenidos de las variables
    xml_file, project y xygraph
Muchas veces en el mismo conjunto de puntos quieres interpolar varias variables
    (o varios proyectos en el mismo fichero xml); puedes colocar todos los
    nombres de proyecto en una lista y ejecutarlos todos a la vez
"""
from time import time
import traceback

from interpol import IDW
import littleLogging as logging

xml_file: str = 'idw.xml'
project: str = 'QCC_tmax_d'
xygraph: bool = True

if __name__ == "__main__":

    try:

        startTime = time()

        for project in ("QCC_pd", "QCC_tmax_d", "QCC_tmin_d"):
            i = IDW(xml_file, project)
            i.idw_ts(xygraph)

        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

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
        print('se ha creado el fichero app.log')
