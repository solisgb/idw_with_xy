# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:19 2019

@author: solis

implementa la interpolación de series temporales en puntos sin datos utilizando
    el método idw
también se puede utilizar para rellenar estaciones; si un punto a interpolar
    coincide en coordenada con un punto con dato, el punto a interpolar tendrá
    el valor del punto con dato -él mismo-
"""
import numpy as np
import littleLogging as logging


class IDW():

    table_name_interpolated_values: str = 'ivalues'
    # paso de tiempo en las series temporales
    time_steps: tuple = ('day', 'lastday_month')
    # separador de columnas en el fichero de resultados
    COL_SEP: str = ';'
    # al interpolar series temporales imprime fecha cada
    PRINT_SECS: float = 5.
    # fichero de puntos en los que se realiza la interpolación
    file_point: str = 'interpolated_points'


    def __init__(self, xml_org: str, project: str):
        """
        xml_org: nombre del fichero xml con los valores de los parámetros
            los contenidos están agrupados por proyectos (label project)
        project: datos del proyecto con el que se va a hacer la interpolación
        """
        self.xml_org = xml_org
        self.project = project
        self._write_params()


    def _write_params(self):
        from datetime import date
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.xml_org)
        root = tree.getroot()
        prj = None
        for element in root.findall('project'):
            if element.get('name') == self.project:
                prj = element
                break
        if not prj:
            raise ValueError(f'No se encuentra el project {self.project}')

        self.description = prj.find('description').text
        self.dbtype = prj.find('dbtype').text
        self.db = prj.find('db').text
        day1 = int(prj.find('day1').text)
        month1 = int(prj.find('month1').text)
        year1 = int(prj.find('year1').text)
        self.datei = date(year1, month1, day1)
        day2 = int(prj.find('day2').text)
        month2 = int(prj.find('month2').text)
        year2 = int(prj.find('year2').text)
        self.datefin = date(year2, month2, day2)
        if self.datei > self.datefin:
            self.datei, self.datefin = self.datefin, self.datei
        self.date_format = '%Y-%m-%d'
        time_step = prj.find('time_step').text
        if time_step not in IDW.time_steps:
            raise ValueError(f'time_step {time_step} debe estar en: ' +\
                             f'{",".join(IDW.time_steps)}')
        self.tstep_type, self.time_step = IDW.__time_step_get(time_step)
        self.no_value = float(prj.find('no_value').text)
        self.float_format = prj.find('float_format').text
        self.variable_short_name = prj.find('variable_short_name').text
        self.ylabel = prj.find('ylabel').text
        self.select = prj.find('select').text
        self.pathin = prj.find('pathin').text
        self.fpoints = prj.find('fpoints').text
        self.skip_lines = int(prj.find('skip_lines').text)
        self.pathout = prj.find('pathout').text
        self.kidw = int(prj.find('kidw').text)
        self.poweridw = float(prj.find('poweridw').text)
        self.epsidw = float(prj.find('epsidw').text)
        if prj.find('xygraph').text.lower().strip() in ('true', '1'):
            self.xygraph = True
        else:
            self.xygraph = False


    def idw_ts(self, xygraph: bool):
        """
        Interpolación de una serie temporal en una serie de puntos sin datos
        xygraph: True si se graba un gráficos con cada punto interpolado
        """
        import sqlite3
        from os.path import join
        from time import time
        from scipy import spatial
        from db_connection import con_get
        import idw

        create_table1 = \
        f"""
        create table if not exists {IDW.table_name_interpolated_values} (
        fid TEXT,
        fecha TEXT,
        value REAL,
        PRIMARY KEY (fid, fecha))"""

        insert1 = \
        f"""
        insert into {IDW.table_name_interpolated_values}
        (fid, fecha, value)
        values(?, ?, ?)"""

        start_time = time()

        # datos donde hay que interpolar, devuelve numpy array of objects
        rows = IDW.__read_file_points(join(self.pathin, self.fpoints),
                                      self.skip_lines)
        # array con los id de los puntos donde se quiere interpolar
        fidi = rows[:, 0]
        # array con las coord. de los puntos
        xi = np.array(rows[:,[1,2]], np.float32)
        # array para los valores interpolados en xi en cada fecha
        zi = np.empty((len(xi)), np.float32)

        # cursor a los datos para hacer las interpolaciones
        con = con_get(self.dbtype, self.db)
        cur = con.cursor()

        # cursor para los valores interpolados
        con1 = sqlite3.connect(':memory:')
        cur1 = con1.cursor()
        cur1.execute(create_table1)

        t0 = IDW.PRINT_SECS
        if self.tstep_type == 2:
            datei = IDW.__month_lastday(self.datei)
        else:
            datei = self.datei

        # los puntos con datos cambian de una fecha a otra por lo que hay que
        # hacer una select para cada fecha; esto hace que el proceso sea
        # sea largo para series temporales largas
        date_nodata = []
        while datei <= self.datefin:
            date_str = datei.strftime(self.date_format)
            if time() - t0 > IDW.PRINT_SECS:
                t0 = time()
                print(date_str)

            # datos en la fecha datei
            cur.execute(self.select, (datei,))
            data = np.array([row for row in cur], np.float32)
            # puede haber fechas sin datos
            if data.size == 0:
                date_nodata.append(date_str)
                logging.append(f'{date_str} no data', False)
                datei = self.__next_date(datei)
                continue

            # builds kdtree tree and set distances
            xydata = data[:,[0,1]]
            tree = spatial.cKDTree(xydata)
            dist, ii = tree.query(xi, k=min(self.kidw, xydata.shape[0]))

            # sort data by distance
            values0 = data[:,2]
            values = np.empty((dist.shape), np.float32)
            idw.sortn(values0, ii, values)

            # idw interpolation
            idw.idwn(self.poweridw, dist, values, self.epsidw, zi)

            # insert interpolated values in sqlite
            for i in range(len(fidi)):
                cur1.execute(insert1, (fidi[i], date_str, float(zi[i])))

            datei = self.__next_date(datei)

        con.close()
        elapsed_time = time() - start_time
        print(f'La interpolación idw tardó {elapsed_time:0.1f} s')

        con1.commit()

        # las fechas sin datos se tratan de interpolar con medianas
        missed = self._fill_with_median(fidi, date_nodata,
                                        con1, cur1, insert1)

        # graba interpolación en ficheros de texto
        self.__write_interpolated_values(cur1)

        # graba fichero con los metadatos de la interpolación
        self.__write_idw_metadata(fidi.size, elapsed_time, missed)

        # graba el fichero de puntos donde se realizó la interpolación
        self.__write_interpolated_points(fidi, xi)

        # graba los gráficos
        if xygraph:
            self.__xy(cur1)


    @staticmethod
    def __xi_get(rows: list) -> np.array:
        """
        extrae el array de coordenadas: su dimensión depende de las de rows
        """
        if rows.shape[1] == 3:  # 2D
            xi = rows[:, [1, 2]].astype(np.float32)
        elif rows.shape[1] == 4:  #3D
            xi = rows[:, [1, 2, 3]].astype(np.float32)
        else:
            raise ValueError('El núm de columnas en el fichero de puntos ' +\
                             'debe 3 o 4: id del punto, x, y, (z)')
        return xi


    def __next_date(self, datei):
        """
        incrementa el valor de date en cada fecha de interpolación
        args:
        datei: obj date a incrementar
        returns
        date object
        """
        if self.tstep_type == 1:
            datei = datei + self.time_step
        elif self.tstep_type == 2:
            datei = IDW.__addmonth_lastday(datei)
        else:
            raise ValueError(f'tstep_type {self.tstep_type} ' +\
                             'no implementado')
        return datei


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
    def __read_file_points(org, skip_lines):
        """
        lee los puntos a interpolar de un fichero de texto y devuelve un
            numpy array
        """
        from os.path import splitext
        _, extension = splitext(org)

        if extension.lower() == '.txt':
            rows = IDW.__read_txt(org, skip_lines)
        elif extension.lower() == '.csv':
            rows = IDW.__read_csv(org, skip_lines)
        else:
            raise ValueError('El fichero de puntos a interpolar tiene ' +\
                             'que tener la extensión txt o csv')
        return rows


    @staticmethod
    def __read_txt(org, skip_lines, separator='\t'):
        """
        leee los puntos a interpolar en formato txt
        """
        with open (org, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i < skip_lines:
                    num = len(line.split(separator))
                    continue
                words = line.strip().split(separator)
                if len(words) == num:
                    lines.append(words)
        return np.array(lines)


    @staticmethod
    def __read_csv(org, skip_lines):
        """
        leee los puntos a interpolar en formato csv
        """
        import csv
        with open(org, newline='') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)
            rows = [row for i, row in enumerate(reader) if i>=skip_lines]
        return np.array(rows)


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
    def __month_lastday(fecha):
        """
        Devuelve el último del mes
        input:
            fecha. date type
        output:
            date type
        """
        from datetime import date
        from calendar import monthrange
        year = fecha.year
        month = fecha.month
        day = monthrange(year, month)[1]
        return date(year, month, day)


    def __write_interpolated_values(self, cur1):
        from os.path import join
        """
        escribe los valores interpolados en el fichero dst
        args
        cur1: cursor a una db sqlite
        """
        select1 = \
        f"""select fid, fecha, value
        from {IDW.table_name_interpolated_values}
        order by fid, fecha"""

        dst = join(self.pathout, self.variable_short_name+'_idw.csv')
        f = open(dst, 'w')
        f.write(f'fid{IDW.COL_SEP}fecha{IDW.COL_SEP}valor\n')
        cadena = '{}' + IDW.COL_SEP + '{}' + IDW.COL_SEP + \
        '{:' + self.float_format + '}\n'
        cur1.execute(select1)
        for row in cur1.fetchall():
            f.write(cadena.format(row[0], row[1], row[2]))
        f.close()


    def __xy(self, cur1):
        """
        graba los gráficos xy de las series interpoladas
        args
        cur1: objeto sqlite cursor
        """
        from datetime import datetime
        from os.path import join

        select = \
        f"""select fid
        from {IDW.table_name_interpolated_values}
        group by fid
        order by fid"""

        select1 = \
        f"""select fecha, value
        from {IDW.table_name_interpolated_values}
        where fid=?
        order by fecha"""

        cur1.execute(select)
        fids = [row[0] for row in cur1.fetchall()]
        for fid in fids:
            fid = str(fid)
            cur1.execute(select1, (fid,))
            data = [[row[0], row[1]] for row in cur1.fetchall()]
            fechas = [datetime.strptime(row[0], self.date_format).date()
                      for row in data]
            values = [row[1] for row in data]
            title = f'Punto interpolado {fid}'
            dst = join(self.pathout, fid+self.variable_short_name+'.png')
            IDW.__xy_ts_plot_1g(title, fechas, values, self.ylabel, dst)


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
        import matplotlib.pyplot as plt
        import matplotlib as mpl

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


    def __write_idw_metadata(self, nfidi: int, elapsed_time: float,
                             missed: list):
        """
        Graba el fichero de metadatos de la interpolación
        args
        nfidi: número de puntos interpolados
        elapsed_time: tiempo transcurrido en segundos
        """
        dst = IDW.__file_name_out(self.pathout, self.fpoints,
                                  self.variable_short_name,
                                  'idw_metadata')
        with open(dst, 'w') as f:
            f.write(f'idw\n')
            f.write(f'Fecha inicial: {self.datei.strftime("%d/%m/%Y")}\n')
            f.write(f'Fecha final: {self.datefin.strftime("%d/%m/%Y")}\n')
            f.write(f'Potencia a la que se eleva la dist, {self.poweridw:f}\n')
            f.write(f'Núm. máx. de datos por interpolación, {self.kidw:d}\n')
            f.write('Distancia entre el punto a interpolar y el valor más ' +\
                    'próximo para asignar el valor al punto, ' +\
                    f'{self.epsidw:f}\n')
            f.write(f'Valor no interpolado, {self.no_value}\n')
            f.write(f'db de los datos, {self.db}\n')
            f.write(f'Núm. de puntos interpolados {nfidi:d}\n')
            f.write(f'Tiempo transcurrido, {elapsed_time:0.1f} s\n')
            if missed:
                f.write(f'No se han podido interpolar\n')
                a = '\n'.join(missed)
                f.write(f'{a}\n')


    def _fill_with_median(self, fidi, date_nodata, con, cur, insert):
        """
        fill not interpolated values using idw in dates date_nodata
        using medians
        """

        select1 = \
        """select value
        from {IDW.table_name_interpolated_values}
        where fid=?
        order by value"""

        select2 = \
        """select value
        from {IDW.table_name_interpolated_values}
        order by value
        """

        if len(date_nodata) == 0:
            return []

        med = np.empty((fidi.size, 12), np.float32)
        med[:] = np.NAN
        missed = []

        for date1 in date_nodata:
            imonth = int(date1[5:7]) - 1
            for i, fidi1 in enumerate(fidi):
                if np.isnan(med[i, imonth]):
                    IDW._median_set(cur, select1, imonth, i, med, fidi1)
                    if np.isnan(med[i, imonth]):
                        IDW._median_set(cur, select2, imonth,
                                        i, med)
                        if np.isnan(med[i, imonth]):
                            logging.append(f'{date1} no median available ' +\
                                           f'for point {fidi1}', False)

                if not np.isnan(med[i, imonth]):
                    cur.execute(insert, (fidi1, date1, float(med[i, imonth])))
                    a = f'point {fidi1}, fecha {date1} interpolated with ' +\
                    'median'
                    logging.append(f'{a}', False)
                else:
                    missed.append(f'point {fidi1}, fecha {date1}')
        con.commit()
        return missed


    @staticmethod
    def _median_set1(cur, select, imonth, i, med, fidi1=None):
        """
        calculates the median using the data retrieved by select
        """
        if fidi1 == None:
            cur.execute(select)
        else:
            cur.execute(select,(fidi1,))
        x = np.array([row[0] for row in cur.fetchall()],
                      np.float32)
        if x.size > 0:
            med[i, imonth] = np.median(x)


    def __write_interpolated_points(self, fidi, xi):
        """
        saves interpolated points in the directory where the interpolated
            results are saved
        """
        from os.path import join
        dst = join(self.pathout,
                   IDW.file_point+'_'+self.variable_short_name+'.csv')
        np.savetxt(dst, np.transpose([fidi, xi[:,0], xi[:,1]]),
                   header='fid,x,y',delimiter=';')


def merge_idw_output_files(orgs: tuple, header: tuple, dst: str,
                           separator: str =';'):
    """
    if you execute idw interpolation for several variables in the same
        date range, you can merge the output files in only one, but only
        in case all dates in date range would have been interpolated
    """
    dst = open(dst, 'w')
    dst.write(separator.join(header) + '\n')

    readers = [open(org1, 'r') for org1 in orgs]
    for i in range(2):
        lines = [ f.readline().strip() for f in readers]

    while lines[0] != '':
        words1 = lines[0].split(separator)[0:3]
        words2 = [line1.split(separator)[2] for line1 in lines[1:]]
        words = words1 + words2
        dst.write(separator.join(words) + '\n')
        lines = [ f.readline().strip() for f in readers]

    for f1 in readers:
        f1.close()
    dst.close()


def output_files_2db(dir_org: str, orgs: dict, dst: str, separator: str =';'):
    """
    export output files to sqlite
    """
    from os.path import join
    import sqlite3

    mydb = 'output_idw.db'

    con = sqlite3.connect(join(dst, mydb))
    cur = con.cursor()

    for k, v in orgs.items():
        print(k)
        cur.execute(f'drop table if exists {k}')
        cur.execute(f'drop table if exists {k}_point')
        cur.execute(f'create table if not exists {k} (' +\
                    'fid TEXT, fecha TEXT, value REAL,' +\
                    'primary key (fid, fecha))')
        cur.execute(f'create table if not exists {k}_point (' +\
                    'fid TEXT, x REAL, y REAL,' +\
                    'primary key (fid))')

        f = open(join(dir_org, v[0]))
        for i, line in enumerate(f):
            if i == 0:
                continue
            w = line.split(separator)
            cur.execute(f'insert into {k}_point (fid, x, y) values(?, ?, ?)',
                        (w[0], float(w[1]), float(w[2])))

        f.close()

        f = open(join(dir_org, v[1]))
        for i, line in enumerate(f):
            if i == 0:
                continue
            w = line.split(separator)
            cur.execute(f'insert into {k} (fid, fecha, value) values(?, ?, ?)',
                        (w[0], w[1], float(w[2])))

        f.close()


    con.commit()
    con.close()
