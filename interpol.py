# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:19 2019

@author: solis

implementa la interpolación de series temporales en puntos sin datos utilizando
    el método idw
también se puede utilizar para rellenar estaciones; si un punto a interpolar
    coincide en coordenada con un punto con dato, el punto a interpolar tendrá
    el valor del punto con dato -él  mismo-
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import spatial
import littleLogging as logging

# paso de tiempo en las series temporales
time_steps = ('day', 'lastday_month')
# al interpolar series temporales imprime fecha cada
PRINT_SECS: float = 5.


class Interpolate():


    def __init__(self, dbtype: str, db: str, select: str,
                 day1: int, month1: int, year1:int,
                 day2: int, month2: int, year2:int,
                 time_step: str, no_value: float,
                 date_format: str, float_format: str):
        from datetime import date

        self.dbtype = dbtype
        self.db = db
        self.select = select
        self.datei = date(year1, month1, day1)
        self.datefin = date(year2, month2, day2)
        if self.datei > self.datefin:
            self.datei, self.datefin = self.datefin, self.datei
        self.tstep_type, self.time_step = \
        Interpolate.__time_step_get(time_step)
        self.no_value = no_value
        self.date_format = date_format
        self.float_format = float_format


    def idw_serie_temporal(self, pathin: str, fpoints: str, skip_lines: int,
                           pathout: str, variable_short_name: str,
                           kidw: int, power: int, epsidw: float,
                           xygraph: bool):
        """
        Método de interpolación inverse distance weighted con distance elevada
            a una potencia, normalmente 2
        Interpola la variable en una serie de puntos dados en 3D. Si no tienes
            valor de Z pon en todos los puntos un valor cte, por ej 0
        Los datos son una serie temporal; los puntos con datos varían en el
            tiempo
        """
        from os.path import join
        import sqlite3
        from time import time
        from db_connection import con_get

        create_table1 = \
        """
        create table if not exists interpolated (
        fid TEXT,
        fecha TEXT,
        value REAL,
        PRIMARY KEY (fid, fecha))"""

        insert1 = \
        'insert into interpolated (fid, fecha, value) values(?, ?, ?);'

        start_time = time()

        con, f = None, None

        # datos donde hay que interpolar
        rows = Interpolate.__points2interpolate_get(join(pathin, fpoints),
                                                    skip_lines)
        fidi = rows[:, 0]  # array con los id de los puntos
        if rows.shape[1] == 3:  # 2D
            xi = rows[:, [1, 2]].astype(np.float32)  # array con las coord.
        elif rows.shape[1] == 4:  #3D
            xi = rows[:, [1, 2, 3]].astype(np.float32)
        else:
            raise ValueError('El núm de columnas en el fichero de puntos ' +\
                             'debe 3 o 4: id del punto, x, y, (z)')
        rows = None
        # array para los valores interpolados
        zi = np.empty((len(xi)), np.float32)

        # cursor a los datos para hacer las interpolaciones
        con = con_get(self.dbtype, self.db)
        cur = con.cursor()

        # cursor para los datos para los gráficos de evolución
        con1 = sqlite3.connect(':memory:')
        cur1 = con1.cursor()
        cur1.execute(create_table1)

        # fichero de salida
        dst = Interpolate.__file_name_out(pathout, fpoints,
                                          variable_short_name, 'idw')
        f = open(dst, 'w')
        f.write('fid\tfecha\tvalor\n')
        cadena = '{}\t{}\t{:' + self.float_format + '}\n'

        t0 = PRINT_SECS
        datei = self.datei
        while datei <= self.datefin:
            date_str = datei.strftime(self.date_format)
            if time() - t0 > PRINT_SECS:
                t0 = time()
                print(date_str)
            cur.execute(self.select, (datei,))
            data = [row for row in cur]
            if xi.shape[1] == 2:
                data = np.array([[row[0], row[1], row[2]] for row in data])
                tree = spatial.cKDTree(data[:, [0, 1]])
            else:
                data = np.array([[row[0], row[1], row[2], row[3]] \
                                  for row in data])
                tree = spatial.cKDTree(data[:, [0, 1, 2]])
            dist, ii = tree.query(xi, k=kidw)

            zi.fill(self.no_value)
            Interpolate.__idwcore(data[:, xi.shape[1]], dist, ii, power,
                                    epsidw, zi)

            for i in range(len(fidi)):
                f.write(cadena.format(fidi[i], date_str, zi[i]))
                cur1.execute(insert1, (fidi[i], date_str, zi[i]))

            if self.tstep_type == 1:
                datei = datei + self.time_step
            elif self.tstep_type == 2:
                datei = Interpolate.__addmonth_lastday(datei)
            else:
                f.close()
                con.close()
                raise ValueError(f'tstep_type {self.tstep_type} ' +\
                                 'no implementado')

        f.close()
        con.close()
        elapsed_time = time() - start_time
        print(f'La interpolación tardó {elapsed_time:0.1f} s')
        self.__write_idw_metadata(pathout, fpoints,
                                  variable_short_name, power,
                                  kidw, epsidw, fidi.size,
                                  elapsed_time)

        self.__xy(pathout, variable_short_name, cur1)


    @staticmethod
    def __time_step_get(time_step: str):
        """
        evalúa el timedelta entre fechas
        """
        from datetime import timedelta
        if time_step == 'day':
            return (1, timedelta(days=1))
        elif time_step == 'lastday_month':
            return (2, None)
        else:
            raise ValueError(f'valor de time_step {time_step} ' +\
                             'no implementado')


    @staticmethod
    def __points2interpolate_get(org, skip_lines):
        """
        lee los puntos a interpolar de un fichero de texto y devuelve un
            numpy array
        """
        with open(org, 'r') as f:
            rows = np.array([row.split('\t')
                             for i, row in enumerate(f.readlines())
                             if i >=skip_lines])
        return rows


    @staticmethod
    def __idwcore(dat, dist, ii, power, epsidw, zi):
        """
        interpolación idw elevado a una potencia
        input
            dat: array 1D de datos dim (n)
            dist: array de distancias dim (n, k)
            ii: índices de los datos de dist en dat
            power: potencia
            zi: array 1D de valores interpolados dim (n)
        output
            Los resultados se graban en zi
        """
        for i in range(zi.shape[0]):
            if dist[i, 0] <= epsidw:
                zi[i] = dat[ii[0]]
                continue
            u = np.array([dat[j] for k, j in enumerate(ii[i])
                         if np.isfinite(dist[i, k])])
            weights = 1. / dist[i, 0:u.size]**power
            weights = weights / np.sum(weights)
            u = u * weights
            zi[i] = np.sum(u)


    @staticmethod
    def __file_name_out(pathout: str, name_file: str, variable_name: str,
                      interpol_method: str):
        """
        forma el nombre del fichero de salida
        """
        from os.path import join, splitext
        name_file, extension = splitext(name_file)
        name_file = f'{name_file}_{variable_name}_{interpol_method}{extension}'
        return join(pathout, name_file)


    @staticmethod
    def __addmonth_lastday(fecha):
        """
        Añade un mesa la fecha y pone al día el último del mes
        ej.: date(1957,1,1:31) -> datedate(1957,2,28)
        input:
            fecha. date type
        output:
            date type
        """
        from datetime import date
        from calendar import monthrange
        year = fecha.year
        month = fecha.month
        if month < 12:
            month += 1
        else:
            year += 1
            month = 1
        day = monthrange(year, month)[1]
        return date(year, month, day)


    @staticmethod
    def __xy(pathout: str, variable_short_name: str, cur1):
        from os.path import join
        """
        cur1: sqlite cursor
        """
        select = \
        'select fid from interpolated group by fid order by fid;'

        select1 = \
        'select fecha, value from interpolated where fid=? order by fecha;'

        cur1.execute(select)
        fids = [row[0] for row in cur1.fetchall()]
        for fid in fids:
            cur1.execute(select1, (fid,))
            data = [[row[0], row[1]] for row in cur1.fetchall()]
            fechas =
            dst = join(pathout, fid+variable_short_name+'.png')


    @staticmethod
    def __xy_ts_plot_1g(title: str, x: list, y: list, ylabel: str, dst: str):
        """
        Dibuja una figura con 1 gráfico (axis) xy
        args
        title: título de la figura
        tsu: lista de objetos Time_series para el gráfico superior
        x: lista de objetos date
        y: lista de valores interpolados float
        dst: nombre fichero destino (debe incluir la extensión png)
        """
        # parámetros específicos
        mpl.rc('font', size=8)
        mpl.rc('axes', labelsize=8, titlesize= 10, grid=True)
        mpl.rc('axes.spines', right=False, top=False)
        mpl.rc('xtick', direction='out', top=False)
        mpl.rc('ytick', direction='out', right=False)
        mpl.rc('lines', linewidth=0.8, linestyle='-', marker='.', markersize=4)
        mpl.rc('legend', fontsize=8, framealpha=0.5, loc='best')

        fig, ax = plt.subplots()

        plt.suptitle(title)
        ax.set_ylabel(ylabel)

        fig.autofmt_xdate()

        ax.plot(x, y)

        fig.savefig(dst)
        plt.close('all')
        plt.rcdefaults()


    def __write_idw_metadata(self, pathout: str, fpoints: str,
                             variable_short_name: str, power: float,
                             kidw: int, epsidw: float, nfidi: int,
                             elapsed_time: float):
        """
        Graba el fichero de metadatos de la interpolación
        """
        dst = Interpolate.__file_name_out(pathout, fpoints,
                                          variable_short_name, 'idw_metadata')
        with open(dst, 'w') as f:
            f.write(f'idw\n')
            f.write(f'potencia a la que se eleva la dist, {power:f}\n')
            f.write(f'núm máx. de datos por interpolación, {kidw:d}\n')
            f.write('distancia entre el punto a interpolar y el valor más ' +\
                    'próximo para asignar el valor al punto, ' +\
                    f'{epsidw:f}\n')
            f.write(f'valor no interpolado, {self.no_value}\n')
            f.write(f'db de los datos, {self.db}\n')
            f.write(f'número de puntos interpolados {nfidi:d}\n')
            f.write(f'tiempo transcurrido, {elapsed_time:0.1f} s\n')
            a = logging.get_as_str()
            if a:
                f.write(f'incidencias\n')
                f.write(f'{a}\n')
