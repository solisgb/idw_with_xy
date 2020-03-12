# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:53:24 2019

@author: solis

Parámetros de un módulo de funciones de interpolación de series temporales
    en puntos sin datos
Implementa idw
Antes de ejecutar el programa lee atentamente los parámetros que intervienen
"""

"""
pathin: directorio donde se ubica en fichero de puntos a interpolar (de la
    forma r'path2dis'; r es un indicador para que se interprete \ como un
    separados de carpetas)
pathout: directorio donde se grabarán los datos interpolados
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
        variable_short_name: nombre muy corto que describe la variable a
        interpolar y se utiliza en el nombre del fichero de resultados
        -por ejemplo pd para precipitación diaria-
El nombre del fichero de resultados se forma: nombre de fpoints +
   variable_short_name + método de interpolación utilizado + extensión.
   También se crea un fichero de metadatos de la interpolación
"""
pathin = r'C:\Users\solis\Documents\work\dpa\balan\solanallosa'
pathout = pathin
fpoints = 'solanallosaxy.txt'
skip_lines: int = 1
variable_short_name = 'pd'

"""=======================DB con los datos====================================
db: base de datos con los datos a interpolar
dbtype: msaccess, sqlite
"""
dbtype = 'postgres'
db = r'bda'

"""============select para extraer los datos de la base de datos===========
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

"""=======fechas entre las que se interpola, ambas incluidas=================
El programa permite interpolar en una serie de puntos na serie temporal entre
    una fecha inicial y una fecha final: la fecha incicial es
    fecha1(day1, month1, year1); la fecha final es fecha2(day2, month2, year2)
"""
day1, month1, year1 = 1, 1, 1985
day2, month2, year2 = 31, 12, 2005

"""==============paso de tiempo para la interpolación========================
A partir de la fecha1, el programa genera una nueva fecha en la que tratará
    de interpolar en los puntos seleccionados; el salto de tiempo entre una
    fecha y la siguiente puede ser:
day. dia a día desde la fecha inicial hasta la fecha final
lastday_month. los datos mensuales están asociados a fechas; cada dato está
    asociado a la última fecha del mes; el programa está preparado para tratar
    estos saltos diarios variables entre fechas -pues no todos los meses
    tienen el mismo número de días-. Si en la bdd los datos mensuales
    no están almacenados de esta manera no encontrará datos para hacer las
    interpolaciones. Es aconsejable que las fechas inicial y final sean
    el último días del mes. ¿Que pasaría si en la fecha inicial el día no fuese
    el último del mes; en esa fecha no encontraría ningún dato y no
    interpolaría, pero la siguiente fecha ya la formaría correctamente el
    último día del mes siguiente.
"""
time_step = 'day'

""" =====================METODOS DE INTERPOLACION========================
poweridw: potencia en el método idw (1/dist**powerid); normalmente 2.
kidw: número de puntos con los que se realiza la interpolación (min 2)
epsidw: si la distancia entre un punto con dato y un punto a interpolar es
    menor que eps, se asigna la distancia del valor más próximo -para
    evitar el infinito al dividir 1/d- siendo d la distancia entre el
    punto a interpolar y el punto con dato si coinciden; si es muy pequeña
    podemos tener desbordamiento numérico
"""
poweridw: float = 2.0
kidw: int = 8
epsidw: float = 0.5

"""=====================OPCIONES GENERALES======================
date_format: formato de fecha en el fichero de salida según las reglas de
    la función strftime de python
float_format: formato valores interpolados
    0.nf indica un número real con n decimales
    f indica un número real con todos sus decimales
no_value: valor para fechas sin dato en las que no se puede interpolar
"""
date_format: str = '%Y-%m-%d'
float_format: str = '0.1f'
no_value: float = -9999.
