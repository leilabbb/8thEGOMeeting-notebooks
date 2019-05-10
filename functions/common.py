#! /usr/bin/env python
import os
import pandas as pd
import requests
import re
import itertools
import time
import xarray as xr
import numpy as np
import datetime as dt
from urllib.request import urlopen
import json
from datetime import timedelta
from geopy.distance import geodesic
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def calculate_mean_pressure(press, ds, refdes, deploy_depth):
    """
    Calculate mean pressure from data, excluding outliers +/- 3 SD
    """
    notes = []
    subsite = refdes.split('-')[1]
    node = refdes.split('-')[1]

    try:
        pressure = ds[press]
        num_dims = len(pressure.dims)
        if len(pressure) > 1:
            # reject NaNs
            p_nonan = pressure.values[~np.isnan(pressure.values)]

            # reject fill values
            p_nonan_nofv = p_nonan[p_nonan != pressure._FillValue]

            # reject data outside of global ranges
            try:
                [pg_min, pg_max] = get_global_ranges(refdes, press)
                if pg_min is not None and pg_max is not None:
                        pgr_ind = reject_global_ranges(p_nonan_nofv, pg_min, pg_max)
                        p_nonan_nofv_gr = p_nonan_nofv[pgr_ind]
                else:
                    p_nonan_nofv_gr = p_nonan_nofv
            except Exception: 
                    print('uFrame is not responding to request for global ranges. Try again later.')
                    p_nonan_nofv_gr = p_nonan_nofv

            if (len(p_nonan_nofv_gr) > 0) and (num_dims == 1):
                [press_outliers, pressure_mean, _, pressure_max, _, _] = variable_statistics(p_nonan_nofv_gr, 3)
                pressure_mean = round(pressure_mean, 2)
                pressure_max = round(pressure_max, 2)
            elif (len(p_nonan_nofv_gr) > 0) and (num_dims > 1):
                print('variable has more than 1 dimension')
                press_outliers = 'not calculated: variable has more than 1 dimension'
                pressure_mean = round(np.nanmean(p_nonan_nofv_gr), 2)
                pressure_max = round(np.nanmax(p_nonan_nofv_gr), 2)
            else:
                press_outliers = None
                pressure_mean = None
                pressure_max = None
                if len(pressure) > 0 and len(p_nonan) == 0:
                    notes.append('Pressure variable all NaNs')
                elif len(pressure) > 0 and len(p_nonan) > 0 and len(p_nonan_nofv) == 0:
                    notes.append('Pressure variable all fill values')
                elif len(pressure) > 0 and len(p_nonan) > 0 and len(p_nonan_nofv) > 0 and len(p_nonan_nofv_gr) == 0:
                    notes.append('Pressure variable outside of global ranges')

        else:  # if there is only 1 data point
            press_outliers = 0
            pressure_mean = round(ds[press].values.tolist()[0], 2)
            pressure_max = round(ds[press].values.tolist()[0], 2)

        try:
            pressure_units = pressure.units
        except AttributeError:
            pressure_units = 'no units attribute for pressure'

        if pressure_mean:
            node = refdes.split('-')[1]
            if ('WFP' in node) or ('MOAS' in subsite):
                pressure_compare = int(round(pressure_max))
            else:
                pressure_compare = int(round(pressure_mean))

            if pressure_units == '0.001 dbar':
                pressure_max = round((pressure_max / 1000), 2)
                pressure_mean = round((pressure_mean / 1000), 2)
                notes.append('Pressure converted from 0.001 dbar to dbar for pressure comparison')
        else:
            pressure_compare = None

        if (not deploy_depth) or (not pressure_mean):
            pressure_diff = None
        else:
            pressure_diff = pressure_compare - deploy_depth

    except KeyError:
        press = 'no seawater pressure in file'
        pressure_diff = None
        pressure_mean = None
        pressure_max = None
        pressure_compare = None
        press_outliers = None
        pressure_units = None
        
    return pressure_compare, pressure_max, pressure_mean


def check_request_status(thredds_url):
    check_complete = thredds_url.replace('/catalog/', '/fileServer/')
    check_complete = check_complete.replace('/catalog.html', '/status.txt')
    session = requests.session()
    r = session.get(check_complete)
    while r.status_code != requests.codes.ok:
        print('Data request is still fulfilling. Trying again in 1 minute.')
        time.sleep(60)
        r = session.get(check_complete)
    print('Data request has fulfilled.')


def compare_lists(list1, list2):
    match = []
    unmatch = []
    for i in list1:
        if i in list2:
            match.append(i)
        else:
            unmatch.append(i)
    return match, unmatch


def compare_datasets(df, mapping, preferred_method, index_i, ds0, ds0_method, ds1, ds1_method):
    
    missing_data_list = []
    diff_gzero_list = []
    var_list = []
    blank_dict = {'missing_data_gaps': [], 'n_missing': [], 'n_missing_days_total': 0,
                                          'n_missing_total': 0}
    for rr in mapping.itertuples():
        index, name_ds0, long_name, name_ds1 = rr
        print(name_ds0, long_name, name_ds1)
        # Compare data from two data streams (round timestamps to the nearest second).
        ds0_rename = '_'.join((str(name_ds0), 'ds0'))
        [ds0_df, ds0_units, n0, n0_nan] = get_ds_variable_info(ds0, name_ds0, ds0_rename)

        ds1_rename = '_'.join((str(name_ds1), 'ds1'))
        [ds1_df, ds1_units, n1, n1_nan] = get_ds_variable_info(ds1, name_ds1, ds1_rename)

        # skip if the variables have more than 1 dimension
        if (type(ds0_df) == str) or (type(ds1_df) == str):
            print('variables have more than 1 dimension')
            continue
        else:
            # Compare units
            if ds0_units == ds1_units:
                unit_test = 'pass'
            else:
                unit_test = 'fail'

            # Merge dataframes from both methods
            merged = pd.merge(ds0_df, ds1_df, on='time', how='outer')

            # Drop rows where both variables are NaNs, and make sure the timestamps are in order
            merged.dropna(subset=[ds0_rename, ds1_rename], how='all', inplace=True)

            if len(merged) == 0:
                print('No valid data to compare')
                n_comparison = 0
                n_diff_g_zero = None
                min_diff = None
                max_diff = None
                ds0_missing_dict = 'No valid data to compare'
                ds1_missing_dict = 'No valid data to compare'
            else:
                merged = merged.sort_values('time').reset_index(drop=True)
                m_intersect = merged[merged[ds0_rename].notnull() & merged[ds1_rename].notnull()]

                # If the number of data points for comparison is less than 1% of the smaller sample size
                # compare the timestamps by rounding to the nearest hour
                if len(m_intersect) == 0 or float(len(m_intersect))/float(min(n0, n1))*100 < 1.00:
                    n_comparison = 0
                    n_diff_g_zero = None
                    min_diff = None
                    max_diff = None

                    utime_df0 = unique_timestamps_hour(ds0)
                    utime_df0['ds0'] = 'ds0'
                    utime_df1 = unique_timestamps_hour(ds1)
                    utime_df1['ds1'] = 'ds1'
                    umerged = pd.merge(utime_df0, utime_df1, on='time', how='outer')
                    umerged = umerged.sort_values('time').reset_index(drop=True)

                    if 'telemetered' in ds0_method:
                        ds0_missing_dict = 'method not checked for missing data'
                    else:
                        ds0_missing = umerged.loc[umerged['ds0'].isnull()]
                        if len(ds0_missing) > 0:
                            ds0_missing_dict = missing_data_times(ds0_missing)
                            if ds0_missing_dict != blank_dict:
                                ds0_missing_dict['n_hours_missing'] = ds0_missing_dict.pop('n_missing')
                                ds0_missing_dict['n_hours_missing_total'] = ds0_missing_dict.pop('n_missing_total')
                            else:
                                ds0_missing_dict = 'timestamps rounded to the hour: no missing data'
                        else:
                            ds0_missing_dict = 'timestamps rounded to the hour: no missing data'

                    if 'telemetered' in ds1_method:
                        ds1_missing_dict = 'method not checked for missing data'
                    else:
                        ds1_missing = umerged.loc[umerged['ds1'].isnull()]
                        if len(ds1_missing) > 0:
                            ds1_missing_dict = cf.missing_data_times(ds1_missing)
                            if ds1_missing_dict != blank_dict:
                                ds1_missing_dict['n_hours_missing'] = ds1_missing_dict.pop('n_missing')
                                ds1_missing_dict['n_hours_missing_total'] = ds1_missing_dict.pop('n_missing_total')
                            else:
                                ds1_missing_dict = 'timestamps rounded to the hour: no missing data'
                        else:
                            ds1_missing_dict = 'timestamps rounded to the hour: no missing data'

                else:
                    # Find where data are available in one dataset and missing in the other if
                    # timestamps match exactly. Don't check for missing data in telemetered
                    # datasets.
                    if 'telemetered' in ds0_method:
                        ds0_missing_dict = 'method not checked for missing data'
                    else:
                        ds0_missing = merged.loc[merged[ds0_rename].isnull()]
                        if len(ds0_missing) > 0:
                            ds0_missing_dict = missing_data_times(ds0_missing)
                            if ds0_missing_dict == blank_dict:
                                ds0_missing_dict = 'no missing data'
                        else:
                            ds0_missing_dict = 'no missing data'

                    if 'telemetered' in ds1_method:
                        ds1_missing_dict = 'method not checked for missing data'
                    else:
                        ds1_missing = merged.loc[merged[ds1_rename].isnull()]
                        if len(ds1_missing) > 0:
                            ds1_missing_dict = missing_data_times(ds1_missing)
                            if ds1_missing_dict == blank_dict:
                                ds1_missing_dict = 'no missing data'
                        else:
                            ds1_missing_dict = 'no missing data'

                    # Where the data intersect, calculate the difference between the methods
                    diff = m_intersect[ds0_rename] - m_intersect[ds1_rename]
                    n_diff_g_zero = sum(abs(diff) > 0.99999999999999999)

                    min_diff = round(min(abs(diff)), 10)
                    max_diff = round(max(abs(diff)), 10)
                    n_comparison = len(diff)
        compare_summary = dict(
                                ds0=dict(name=name_ds0, units=ds0_units, n=n0, n_nan=n0_nan, missing=ds0_missing_dict),
                                ds1=dict(name=name_ds1, units=ds1_units, n=n1, n_nan=n1_nan, missing=ds1_missing_dict),
                                unit_test=unit_test, n_comparison=n_comparison, n_diff_greater_zero=n_diff_g_zero,
                                min_abs_diff=min_diff, max_abs_diff=max_diff
                              )

        name = compare_summary[preferred_method]['name']
        units = compare_summary[preferred_method]['units']
        unit_test = compare_summary['unit_test']
        n = compare_summary[preferred_method]['n']
        n_nan = compare_summary[preferred_method]['n_nan']
        missing_data = compare_summary[preferred_method]['missing']
        n_comparison = compare_summary['n_comparison']
        min_abs_diff = compare_summary['min_abs_diff']
        max_abs_diff = compare_summary['max_abs_diff']
        n_diff_greater_zero = compare_summary['n_diff_greater_zero']
        if n_comparison > 0:
            percent_diff_greater_zero = round((float(n_diff_greater_zero)/float(n_comparison) * 100), 2)
        else:
            percent_diff_greater_zero = None

        missing_data_list.append(str(missing_data))
        diff_gzero_list.append(percent_diff_greater_zero)
        var_list.append(name) 

        df0 = pd.DataFrame({'name':[name], 
        'unit': [units], 
        'unit_test': [unit_test], 
        'n': [n] ,
        'n_nan': [n_nan] ,
        'missing_data': [missing_data], 
        'n_comparison': [n_comparison], 
        'min_abs_diff': [min_abs_diff], 
        'max_abs_diff': [max_abs_diff],
        'n_diff_greater_zero': [n_diff_greater_zero], 
        'percent_diff_greater_zero': [percent_diff_greater_zero]}, index= [index_i])
        df = df.append(df0)
        
    return df, missing_data_list, diff_gzero_list, var_list

def create_dir(new_dir):
    # Check if dir exists.. if it doesn't... create it.
    if not os.path.isdir(new_dir):
        try:
            os.makedirs(new_dir)
        except OSError:
            if os.path.exists(new_dir):
                pass
            else:
                raise


def deploy_location_check(refdes):
    # Calculate the distance in kilometers between an instrument's location (defined in asset management) and previous
    # deployment locations
    deploy_loc = {}
    dr_data = refdes_datareview_json(refdes)
    for i, d in enumerate(dr_data['instrument']['deployments']):
        deploy_loc[i] = {}
        deploy_loc[i]['deployment'] = d['deployment_number']
        deploy_loc[i]['lat'] = d['latitude']
        deploy_loc[i]['lon'] = d['longitude']

    # put info in a data frame
    df = pd.DataFrame.from_dict(deploy_loc, orient='index').sort_index()
    y = {}
    for i, k in df.iterrows():
        if i > 0:
            loc1 = [k['lat'], k['lon']]
            d1 = int(k['deployment'])
            for x in range(i):
                info0 = df.iloc[x]
                compare = 'diff_km_D{}_to_D{}'.format(d1, int(info0['deployment']))
                loc0 = [info0['lat'], info0['lon']]
                diff_loc = round(geodesic(loc0, loc1).kilometers, 4)
                y.update({compare: diff_loc})
    return y


def eliminate_common_variables(list):
    # time is in this list because it doesn't show up as a variable in an xarray ds
    common = ['quality_flag', 'provenance', 'id', 'deployment', 'obs', 'lat', 'lon']
    regex = re.compile(r'\b(?:%s)\b' % '|'.join(common))
    list = [s for s in list if not regex.search(s)]
    return list


def filter_collocated_instruments(main_sensor, datasets):
    # Remove collocated instruments from a list of datasets from THREDDS
    datasets_filtered = []
    for d in datasets:
        fname = d.split('/')[-1]
        if main_sensor in fname:
            datasets_filtered.append(d)
    return datasets_filtered


def filter_other_streams(r, stream_list, fdatasets):
    # Remove other streams from a list of datasets from THREDDS
    datasets_filtered = []
    for d in fdatasets:
        fname = d.split(r + '-')[-1].split('_2')[0]
        for s in stream_list:
            if s == fname:
                if d not in datasets_filtered:
                    datasets_filtered.append(d)


    return datasets_filtered


def format_dates(dd):
    fd = dt.datetime.strptime(dd.replace(',', ''), '%m/%d/%y %I:%M %p')
    fd2 = dt.datetime.strftime(fd, '%Y-%m-%dT%H:%M:%S')
    return fd2


def found_data_in_another_stream(missing_data_list):
    # Check if data are found in a "non-preferred" stream for any science variable
    md_unique = np.unique(missing_data_list).tolist()
    md_options = ['timestamp_seconds do not match']
    if len(md_unique) == 0:
        fd_test = 'no other streams for comparison'
    elif len(md_unique) == 1 and 'no missing data' in md_unique[0]:
        fd_test = 'pass'
    elif len(md_unique) == 1 and md_unique[0] in md_options:
        fd_test = 'no comparison: timestamps do not match'
    elif len(md_unique) == 1 and md_unique[0] in 'No valid data to compare':
        fd_test = 'No valid data to compare'
    else:
        n_missing_gaps = []
        n_missing_days = []
        for md in md_unique:
            if 'no missing data' in md:
                continue
            elif 'No valid data to compare' in md:
                continue
            else:
                md = ast.literal_eval(md)
                n_missing_gaps.append(len(md['missing_data_gaps']))
                n_missing_days.append(md['n_missing_days_total'])
        if len(n_missing_gaps) == 0:
            fd_test = 'pass'
        else:
            n_missing_gaps = np.unique([np.amin(n_missing_gaps), np.amax(n_missing_gaps)]).tolist()
            n_missing_days = np.unique([np.amin(n_missing_days), np.amax(n_missing_days)]).tolist()

            fd_test = 'fail: data found in another stream (gaps: {} days: {})'.format(n_missing_gaps, n_missing_days)
    return fd_test

def found_data_in_another_stream_diff(fd_test, diff_gzero_list, var_list):
    # Check that the difference between multiple methods for science variables is less than 0
    comparison_details = dict()
    if fd_test == 'No valid data to compare':
        comparison_details = 'No valid data to compare'
        comparison_test = 'No valid data to compare'
    else:
        if len(diff_gzero_list) > 0:
            if list(set(diff_gzero_list)) == [None]:
                comparison_details = 'no comparison: timestamps do not match'
                comparison_test = 'no comparison: timestamps do not match'
            else:
                compare_check = [100.00 - dgz for dgz in diff_gzero_list if dgz is not None]
                comparison_details, ilst = group_percents(comparison_details, compare_check)
                if len(ilst) > 0:
                    vars_fail = [str(var_list[i]) for i in ilst]
                    comparison_test = 'fail: check {}'.format(vars_fail)
                else:
                    comparison_test = 'pass'
        else:
            comparison_details = 'no other streams for comparison'
            comparison_test = 'no other streams for comparison'
            
    return comparison_details

def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def get_global_ranges(refdes, variable, api_user=None, api_token=None):
    port = '12578'
    spl = refdes.split('-')
    base_url = '{}/qcparameters/inv/{}/{}/{}/'.format(port, spl[0], spl[1], '-'.join((spl[2], spl[3])))
    url = 'https://ooinet.oceanobservatories.org/api/m2m/{}'.format(base_url)
    if (api_user is None) or (api_token is None):
        r = requests.get(url, verify=False)
    else:
        r = requests.get(url, auth=(api_user, api_token), verify=False)

    if r.status_code is 200:
        if r.json():  # If r.json is not empty
            values = pd.io.json.json_normalize(r.json())
            t1 = values[values['qcParameterPK.streamParameter'] == variable]
            if not t1.empty:
                t2 = t1[t1['qcParameterPK.qcId'] == 'dataqc_globalrangetest_minmax']
                if not t2.empty:
                    global_min = float(t2[t2['qcParameterPK.parameter'] == 'dat_min'].iloc[0]['value'])
                    global_max = float(t2[t2['qcParameterPK.parameter'] == 'dat_max'].iloc[0]['value'])
                else:
                    global_min = None
                    global_max = None
            else:
                global_min = None
                global_max = None
        else:
            global_min = None
            global_max = None
    else:
        raise Exception('uFrame is not responding to request for global ranges. Try again later.')
        global_min = None
        global_max = None
    return [global_min, global_max]


def get_ds_variable_info(dataset, variable_name, rename):
    ds_units = var_units(dataset[variable_name])
    if len(dataset[variable_name].dims) > 1:
        print('variable has more than 1 dimension')
        ds_df = '>1 dim'
        n = None
        n_nan = None
    else:
        ds_df = pd.DataFrame({'time': dataset['time'].values, variable_name: dataset[variable_name].values})
        ds_df.rename(columns={str(variable_name): rename}, inplace=True)
        n = len(ds_df[rename])
        n_nan = sum(ds_df[rename].isnull())

        # round to the nearest second
        ds_df['time'] = ds_df['time'].map(lambda t: t.replace(microsecond=0) + timedelta(seconds=(round(t.microsecond / 1000000.0))))

    return [ds_df, ds_units, n, n_nan]



def get_nc_urls(catalog_urls):
    """
    Return a list of urls to access netCDF files in THREDDS
    :param catalog_urls: List of THREDDS catalog urls
    :return: List of netCDF urls for data access
    """
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
    datasets = []
    for i in catalog_urls:
        # check that the request has fulfilled
        check_request_status(i)

        dataset = requests.get(i).text
        ii = re.findall(r'href=[\'"]?([^\'" >]+)', dataset)
        x = re.findall(r'(ooi/.*?.nc)', dataset)
        for i in x:
            if i.endswith('.nc') == False:
                x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        dataset = [os.path.join(tds_url, i) for i in x]
        datasets.append(dataset)
    datasets = list(itertools.chain(*datasets))
    return datasets


def get_preferred_stream_info(refdes):
    ps_link = 'https://raw.githubusercontent.com/ooi-data-lab/data-review-tools/master/data_review/output/{}/{}/{}-preferred_stream.json'.format(
        refdes.split('-')[0], refdes, refdes)
    pslnk = urlopen(ps_link)
    psl = json.loads(pslnk.read())
    ps_df = pd.DataFrame.from_dict(psl, orient='index')
    ps_df = ps_df.reset_index()
    ps_df.rename(columns={'index': 'deployment'}, inplace=True)
    ps_df.sort_values(by=['deployment'], inplace=True)
    n_streams = len(ps_df.columns) - 1

    return ps_df, n_streams


def get_url_content(url_address):
    """
    Return content of a url in a json format
    """
    r = requests.get(url_address)
    if r.status_code is not 200:
        print(r.reason)
        print('Problem wi chatth', url_address)
    else:
        url_content = r.json()
    return url_content
 

def group_percents(summary_dict, lst):
    percent_grps = [99, [95, 99], [75, 95], [50, 75], [25, 50], 25]
    for grp in percent_grps:
        if grp == 99:
            x99 = []
            ilst = []
            for i, x in enumerate(lst):
                if type(x) is not str and x >= grp:
                    x99.append(x)
                elif type(x) is not str and x < grp:
                    ilst.append(i)
            #x99 = len([x for x in lst if (type(x) is not str and x > grp)])
            if len(x99) > 0:
                summary_dict['99'] = len(x99)
        elif grp == 25:
            x0 = len([x for x in lst if x < grp])
            if x0 > 0:
                summary_dict['0'] = x0
        else:
            xgrp = len([x for x in lst if grp[0] <= x < grp[1]])
            if xgrp > 0:
                summary_dict[str(int(grp[0]))] = xgrp
    return summary_dict, ilst


def insert_into_dict(d, key, value):
    if key not in d:
        d[key] = [value]
    else:
        d[key].append(value)
    return d


def long_names(dataset, vars):
    name = []
    long_name = []
    for v in vars:
        name.append(v)  # list of recovered variable names

        try:
            longname = dataset[v].long_name
        except AttributeError:
            longname = vars

        long_name.append(longname)

    return pd.DataFrame({'name': name, 'long_name': long_name})


def missing_data_times(df):
    # return a dictionary of time ranges, number of data points and number of days where data are missing (but available
    # in a comparable dataset). skips gaps that are only 1 data point (or one hour if data are rounded to the hour).
    md_list = []
    n_list = []
    mdays = []
    index_break = []
    ilist = df.index.tolist()

    if len(ilist) == 1:
        ii = ilist[0]
        md_list.append(pd.to_datetime(str(df['time'][ii])).strftime('%Y-%m-%dT%H:%M:%S'))
        n_list.append(1)
    else:
        for i, n in enumerate(ilist):
            if i == 0:
                index_break.append(ilist[i])
            elif i == (len(ilist) - 1):
                index_break.append(ilist[i])
            else:
                if (n - ilist[i-1]) > 1:
                    index_break.append(ilist[i-1])
                    index_break.append(ilist[i])

        for ii, nn in enumerate(index_break):
            if ii % 2 == 0:  # check that the index is an even number
                if index_break[ii + 1] != nn:  # only list gaps that are more than 1 data point
                    try:
                        # create a list of timestamps for each gap to get the unique # of days missing from one dataset
                        time_lst = [df['time'][t].date() for t in range(nn, index_break[ii + 1] + 1)]
                    except KeyError:  # if the last data gap is only 1 point, skip
                        continue
                    md_list.append([pd.to_datetime(str(df['time'][nn])).strftime('%Y-%m-%dT%H:%M:%S'),
                                    pd.to_datetime(str(df['time'][index_break[ii + 1]])).strftime('%Y-%m-%dT%H:%M:%S')])
                    n_list.append(index_break[ii + 1] - nn + 1)
                    mdays.append(len(np.unique(time_lst)))

    n_total = sum(n_list)
    n_days = sum(mdays)

    return dict(missing_data_gaps=md_list, n_missing=n_list, n_missing_total=n_total, n_missing_days_total=n_days)


def nc_attributes(nc_file):
    """
    Return global information from a netCDF file
    :param nc_file: url for a netCDF file on the THREDDs server
    """
    with xr.open_dataset(nc_file) as ds:
        fname = nc_file.split('/')[-1].split('.nc')[0]
        subsite = ds.subsite
        node = ds.node
        sensor = ds.sensor
        refdes = '-'.join((subsite, node, sensor))
        method = ds.collection_method
        stream = ds.stream
        deployment = fname[0:14]

    return fname, subsite, refdes, method, stream, deployment


def refdes_datareview_json(refdes):
    """
    Returns information about a reference designator from the Data Review Database
    """
    url = 'http://datareview.marine.rutgers.edu/instruments/view/'
    ref_des_url = os.path.join(url, refdes)
    ref_des_url += '.json'
    r = requests.get(ref_des_url).json()
    return r


def reject_extreme_values(data):
    """
    Reject extreme values
    """
    return (data > -1e7) & (data < 1e7)


def reject_global_ranges(data, gmin, gmax):
    """
    Reject data outside of global ranges
    """
    return (data >= gmin) & (data <= gmax)


def reject_outliers(data, m=3):
    """
    Reject outliers beyond m standard deviations of the mean.
    :param data: numpy array containing data
    :param m: number of standard deviations from the mean. Default: 3
    """
    stdev = np.nanstd(data)
    if stdev > 0.0:
        ind = abs(data - np.nanmean(data)) < m * stdev
    else:
        ind = len(data) * [True]

    return ind


def return_array_subsites_standard_loc(array):
    DBurl= 'https://datareview.marine.rutgers.edu/regions/view/{}.json'.format(array)
    url_ct = get_url_content(DBurl)['region']['sites']
    loc_df = pd.DataFrame()
    for ii in range(len(url_ct)):
        if url_ct[ii]['reference_designator'] != 'CP05MOAS':
            data = {
                    'lat': url_ct[ii]['latitude'],
                    'lon': url_ct[ii]['longitude'],
                    'max_z': url_ct[ii]['max_depth']
                    }
            new_r = pd.DataFrame(data, columns=['lat', 'lon', 'max_z'], index=[url_ct[ii]['reference_designator']])
            loc_df = loc_df.append(new_r)
    return loc_df


def return_raw_vars(ds_variables):
    """
    Return a list of raw variables (eliminating engineering, qc, and timestamps)
    """
    misc_vars = ['quality', 'string', 'timestamp', 'deployment', 'provenance', 'qc', 'time', 'mission', 'obs', 'id',
                 'serial_number', 'volt', 'ref', 'sig', 'amp', 'rph', 'calphase', 'phase', 'checksum', 'description',
                 'product_number']
    reg_ex = re.compile('|'.join(misc_vars))
    raw_vars = [s for s in ds_variables if not reg_ex.search(s)]
    return raw_vars


def return_science_vars(stream):
    """
    Return only the science variables (defined in preload) for a data stream
    """
    sci_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        if p['data_product_type'] == 'Science Data':
            sci_vars.append(p['name'])
    return sci_vars


def return_stream_vars(stream):
    """
    Return all variables that should be found in a stream (from the data review database)
    """
    stream_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        stream_vars.append(p['name'])
    return stream_vars


def stream_word_check(method_stream_dict):
    """
    Check stream names for cases where extra words are used in the names
    """
    omit_word = ['_dcl', '_imodem', '_conc']
    mm = []
    ss = []
    ss_new = []

    for y in method_stream_dict.keys():
        mm.append(str(y).split('-')[0])
        ss.append(str(y).split('-')[1])

    for s in ss:
        wordi = []
        for word in omit_word:
            if word in s:
                wordi.append(word)
                break

        if wordi:
            fix = s.split(wordi[0])
            if len(fix) == 2:
                ss_new.append(fix[0] + fix[1].split('_recovered')[0])
        elif '_recovered' in s:
            ss_new.append(s.split('_recovered')[0])

        else:
            ss_new.append(s)
    return pd.DataFrame({'method': mm, 'stream_name': ss, 'stream_name_compare': ss_new})


def timestamp_gap_test(df):
    gap_list = []
    df['diff'] = df['time'].diff()
    index_gap = df['diff'][df['diff'] > pd.Timedelta(days=1)].index.tolist()
    for i in index_gap:
        gap_list.append([pd.to_datetime(str(df['time'][i-1])).strftime('%Y-%m-%dT%H:%M:%S'),
                         pd.to_datetime(str(df['time'][i])).strftime('%Y-%m-%dT%H:%M:%S')])
    return gap_list


def validate_sci_var_report(rd, sv, ds, index, valid_list_index):
    try:
        var = ds[sv]
        vnum_dims = len(var.dims)
        if vnum_dims > 2:
            print('variable has more than 2 dimensions')
            num_outliers = None
            mean = None
            vmin = None
            vmax = None
            sd = None
            n_stats = 'variable has more than 2 dimensions'
            var_units = var.units
            n_nan = None
            n_fv = None
            n_grange = None
            fv = None
            n_all = None
        else:
            if vnum_dims > 1:
                n_all = [len(var), len(var.values.flatten())]
            else:
                n_all = len(var)
            n_nan = int(np.sum(np.isnan(var.values)))
            fv = var._FillValue
            var_nofv = var.where(var != fv)
            n_fv = int(np.sum(np.isnan(var_nofv.values))) - n_nan

            # reject data outside of global ranges
            [g_min, g_max] = get_global_ranges(rd, sv)
            if g_min is not None and g_max is not None:
                var_gr = var_nofv.where((var_nofv >= g_min) & (var_nofv <= g_max))
                n_grange = int(np.sum(np.isnan(var_gr)) - n_fv - n_nan)
            else:
                n_grange = 'no global ranges'
                var_gr = var_nofv

            if vnum_dims == 1:
                if list(np.unique(np.isnan(var_gr.values))) != [True]:
                    [num_outliers, mean, vmin, vmax, sd, n_stats] = variable_statistics(var_gr, 5)
                else:
                    num_outliers = None
                    mean = None
                    vmin = None
                    vmax = None
                    sd = None
                    n_stats = 0
            else:
                num_outliers = None
                mean = None
                vmin = None
                vmax = None
                sd = None
                n_stats = None
            var_units = var.units

    except KeyError:
        num_outliers = None
        mean = None
        vmin = None
        vmax = None
        sd = None
        n_stats = 'variable not found in file'
        var_units = None
        n_nan = None
        n_fv = None
        fv = None
        n_grange = None
        n_all = None
    
    if type(n_stats) == str:
        percent_valid_data = 'stats not calculated'
    elif type(n_all) == list:
        if type(n_gr) == str:
            n1 = n_all[1] - n_nan - n_fv
        else:
            n1 = n_all[1] - n_nan - n_fv - n_gr
        percent_valid_data = round((float(n1) / float(n_all[1]) * 100), 2)
    else:
        percent_valid_data = round((float(n_stats)/float(n_all) * 100), 2)
    
    valid_list_index.append(percent_valid_data)
    pvd_test = dict()
    snc = len([x for x in valid_list_index if x == 'stats not calculated'])
    if snc > 0:
        pvd_test['stats not calculated'] = snc
    else:
        valid_list_index = [round(v) for v in valid_list_index]
        pvd_test, dlst = group_percents(pvd_test, valid_list_index)
    
        
    if vnum_dims > 1:
        sv = '{} (dims: {})'.format(sv, list(var.dims))
    else:
        sv = sv   
                             
    valid_sci_dic = pd.DataFrame({
                                  'sv': [sv], 'var_units': [var_units], 'fv:fill_value': [fv],
                                  'n_fv': [n_fv],'gr: global_range': [[g_min, g_max]],'n_gr':[n_grange],
                                  'n_nan': [n_nan],  'num_outliers': [num_outliers],
                                  'n_stats': [n_stats], 'mean': [mean],'vmin': [vmin],
                                  'vmax': [vmax], 'sd': [sd], 'percent_valid_data': [percent_valid_data], 
                                  'pvd_test': [pvd_test], 'dlst': [dlst]
                                 }, index = [index])
                                 
    return valid_sci_dic


def variable_statistics(var_data, stdev=None):
    """
    Calculate statistics for a variable of interest
    :param variable: array containing data
    :param stdev: desired standard deviation to exclude from analysis
    """
    if stdev is None:
        varD = var_data
        num_outliers = None
    else:
        ind = reject_extreme_values(var_data)
        var = var_data[ind]

        ind2 = reject_outliers(var, stdev)
        varD = var[ind2]
        varD = varD.astype('float64')  # force variables to be float64 (float32 is not JSON serializable)
        num_outliers = len(var_data) - len(varD)

    mean = round(np.nanmean(varD), 4)
    min = round(np.nanmin(varD), 4)
    max = round(np.nanmax(varD), 4)
    sd = round(np.nanstd(varD), 4)
    n = len(varD)

    return [num_outliers, mean, min, max, sd, n]


def var_units(variable):
    try:
        y_units = variable.units
    except AttributeError:
        y_units = 'no_units'

    return y_units