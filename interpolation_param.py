# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:53:24 2019

@author: solis

Parámetros de un módulo de funciones de interpolación de series temporales
    en puntos sin datos
Implementa idw
Antes de ejecutar el programa lee atentamente los parámetros que intervienen
"""

"""=====================PARÁMETROS GENERALES===============================
Necesarios para instanciar una clase Interpol
dbtype: msaccess, sqlite, puede ser ms_access, sqlite o postgres
db: base de datos con los datos a interpolar
select: select de la que se extraen los datos
day1, month1, year1: fecha inicial de la importación
day2, month2, year2: fecha final de la importación
time_step: paso de tiempo de la serie a interpolar: day o lastday_month
no_value: valor para fechas sin dato en las que no se puede interpolar
float_format: formato del valor interpolado: por ej. 0.0f, 0.1f...
variable_short_name: nombre muy corto de la variable
ylabel: etiqueta y en el gráfico xy de las series interpoladas
"""
dbtype = 'postgres'
db = r'bda'
# =============================================================================
# SELECT1 = "SELECT Estaciones.ID, Estaciones.X, Estaciones.Y, " +\
#               "Estaciones.ALTITUD as Z, PD.Fecha, PD.VALUE as v " +\
#           "FROM Estaciones INNER JOIN PD ON Estaciones.ID = PD.ID " +\
#           "WHERE PD.Fecha=?;"
# =============================================================================
select = \
"""
select e.x_utm , e.y_utm , p.precip
from met.cl_est_climat e inner join met.cl_precip_diaria p
    on e.c_clima = p.c_clima
where p.metodo_i = 1 and p.fh_medida = %s;
"""
day1, month1, year1 = 1, 1, 1985
day2, month2, year2 = 31, 12, 2015
time_step = 'day'
no_value: float = -9999.
float_format: str = '0.1f'
variable_short_name = 'pd'
ylabel: str = 'Pd dmm'


""" ===========PARÁMETROS RELACIONADOS CON LA INTERPOLACIÓN=================
pathin: directorio donde se ubica en fichero de puntos a interpolar (de la
    forma r'path2dis'; r es un indicador para que se interprete \ como un
    separados de carpetas)
fpoints. Fichero con los puntos a interpolar y de resultados. El fichero debe
    tener al menos 1 fila y 3 o 4 columnas de datos:
        columna 1 el código del punto
        columnas 2 y 3: coordenadas x e y
        columna 4 (opcional): coordenada z
    El separador decimal es el punto, por ej. 1.2 y no debe haber separadores
        de miles, por ej. 1000.3. El separador de columnas es el tabulador.
    skip_lines: número de lineas en el fichero de puntos a interpolar que no
        se leen (cabecera, normalmente 1)
    Si el fichero tiene 4 columnas el programa entiende que es una
    interpolación 3D; si tiene 2 columnas será una interpolación en 2D
    Compara las distancias en horizontal entre puntos con datos próximos y
        el gradiente del alturas; si no es necesario no hagas interpolación 3D
skip_lines: número de lineas en el fichero de puntos que no se leen (cabecera)
pathout: directorio donde se grabarán los datos interpolados
kidw: número de puntos con los que se realiza la interpolación (min 2)
poweridw: potencia en el método idw (1/dist**powerid); normalmente 2.
epsidw: si la distancia entre un punto con dato y un punto a interpolar es
    menor que eps, se asigna la distancia del valor más próximo
xygraph: se graban los gráficos de las series interpoladas: True o False
"""

pathin = r'C:\Users\solis\Documents\work\dpa\balan\solanallosa'
fpoints = 'solanallosaxy.txt'
skip_lines: int = 1
pathout = pathin
kidw: int = 8
poweridw: float = 2.0
epsidw: float = 0.5
xygraph: bool = True


"""============SELECT PARA EXTRAER LOS DATOS DE LA BASE DE DATOS===========
La select deve devolver las siguientes columnas
Para interpolación en 3D
    x, y, z, valor de la variable
    los nombres de las columnas son indiferentes
Para interpolar en 2D la columna z no va; no obstante, si el fichero de puntos
    a interpolar es 2D, no pasa nada si se incluye el valor de z, aunque no se
    utiliza
La select debe estar construida con un parámetro fecha en el where, pues la
    select se ejecuta para cada fecha de la serie temporal; para ms access
    o sqlite el parámetro se indica con una ? -por ejemplo columna_fecha=?-
La determinación de si la interpolación va a ser en 2D o en 3D va estar
     definida por las columnas del fichero de puntos a interpolar
"""

"""================FICHERO DE RESULTADOS===================================
El nombre del fichero de resultados se forma: nombre de fpoints +
   variable_short_name + método de interpolación utilizado + extensión.
   También se crea un fichero de metadatos de la interpolación
"""
