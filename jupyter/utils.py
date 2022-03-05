import subprocess
import os, glob, io
import shutil

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import itertools
from itertools import repeat

import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import random, string
import uuid

from zipfile import ZipFile

import numpy as np
import pandas as pd 
import cv2
import json
import math
import pickle
# import paramiko
from pyorthanc import Orthanc, RemoteModality

import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_windowing, apply_modality_lut,dtype_corrected_for_endianness
from pydicom.pixel_data_handlers.numpy_handler import get_pixeldata
from pydicom.pixel_data_handlers.util import apply_color_lut



####################################
### MAIN FUNCTIONS FOR FILTERING ###
####################################


def checkpointer(fn, sequence, batch_size,
                dump_file='checkpoint_data_dump.json',
                checkpoint=0):
    # Repeatedly attempts to run the function on batch_size subsections of the sequence. If the function throws an execption, the outputs so far from the function will be saved along with the current state. The function will be stopped, then restarted from the last saved state. 
    sequence_length = len(sequence)
    n_iter = math.ceil(sequence_length/batch_size)
    output_array = []
    # now_string = datetime.now().strftime("%Y_%m_%d-%H:%M")
    dump_file = 'checkpoint_data_dump.json'

    print("Running function: '{}' in checkpointer".format(fn.__name__))
    print("Elements in sequence: {}".format(sequence_length))
    print("Number of iterations required: {}".format(n_iter))
    while checkpoint < n_iter:
        print("\nStarting at checkpoint {}".format(checkpoint))
        
        # Load checkpoint data if available
        if checkpoint != 0:    
            try:
                with open(dump_file,'r') as f:
                    output_array = json.load(f)
                    print("Loading {} elements from {} file".format(
                        len(output_array), dump_file))
            except:
                print("No data to preload")
                pass
        
        # Iterate through sequence using batch-sizes. If an exception is 
        # raised in 'fn', dump the data to the checkpoint_data_dump file.    
        for subgroup in itertools.islice(get_subgroup(sequence, batch_size),
                checkpoint, None):
            try: 
                fn_output = fn(subgroup)
                # print(fn_output)
                output_array.extend(fn_output)
                checkpoint = checkpoint+1
            except:
                # print(Exception)
                print("System crashed at checkpoint {}".format(checkpoint))
                print("Saving data to {}".format(dump_file))
                with open(dump_file,'w') as f:
                    json.dump(output_array,f)
                break

    return output_array

def months_between_df(dates):
    """
    Calcualtes number of months between two dates in an list.
    In a dataframe, create a column of ['PatientBirthDate', 'StudyDate']
    and apply this function to that column to create a new column.
    Inputs:
      - dates [list]: List of date strings. The first element should be 
      chronologically earlier than the second element.
    Outputs:
      - months_between [int]: Number of months between the two dates in the 
      input list.
    Usage:
        datePairs = list(zip(studies_df['PatientBirthDate'],
                             studies_df['StudyDate']))
        studies_df['datePairs'] = datePairs
        studies_df['PatientAgeMonths'] = studies_df.datePairs.apply(
            lambda x: months_between_df(x)
        )
    """
    d1, d2 = dates
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return round(abs((d2 - d1).days)/(365.25 / 12))

def save_zip(z,targetdir):
    file_like_object = io.BytesIO(z)
    with ZipFile(file_like_object,"r") as zip_ref:
        zip_ref.extractall(targetdir)
    file_like_object.close()

def get_subgroup(sequence, size):
    """
    Returns a generator which splits a sequence into subgroups of a given size. Enables iteration over subgroups to prevent crashes resulting in complete loss of data.
    Inputs:
      - sequence [iterable]: Iterable object containing the sequence to split 
        into subgroups
      - size [int]: Size of subgroups to be split into.
    Outputs:
      - subgroups [generator]
    """
    return (sequence[pos:pos+size] for pos in range(0, len(sequence), size))

def create_dateRanges(start,end,nbins=2):
    '''
    Creates a list of datarange strings to use for dtabase querying
    Inputs: 
      - start (pd.Timestamp() object): First timepoint in range (inclusive).
      - end (pd.Timestamp() object): Final timepoint in range (exclusive).
      - nbins (Int): Number of subragnes to create.
     Outputs:
      - start [list]: List strings representing of equally sized date-ranges
                      covering total date range [start, end).
     Usage:
        start = pd.Timestamp('20000101')
        end = pd.Timestamp('20201201')
        dateranges = create_dateRanges(start, end, 10)
    '''
    t = np.linspace(start.value, end.value, nbins+1)
    t = pd.to_datetime(t, errors='coerce').round('D')
    # t2 = t1 - pd.to_timedelta(t1, unit='d')
    t = pd.DataFrame(t,columns=['Date1'])
    t['Day'] = 1

    t['Date2'] = t['Date1'] -  pd.to_timedelta(t['Day'], unit='d')

    t = pd.concat([t['Date1'][:-1].reset_index(),t['Date2'][1:].reset_index()],axis=1)
    t = t[['Date1','Date2']]
    t['DateRange'] = t.Date1.apply(lambda x: x.strftime('%Y%m%d')) +'-' +t.Date2.apply(lambda x: x.strftime('%Y%m%d')) 

    dateRanges = list(t['DateRange'])
    return dateRanges

def studieslist2df(studiesList):
    '''
    Converts a list of multiple study queries into a single dataframe. 
    This function safely handles lists containing empty elements.
    Inputs: 
      - studiesList [list]: List of study lists. Each study list could be the
            output of a single 'queryStudies' search with different search terms. Each study list must then be combined into a final list which is the input to this function. 
     Outputs:
      - studies_df [dataframe]: Dataframe containing all studies. 
     Usage:
        studies_CT = queryStudies(orthanc, PACS, StudyDate=date_ranges,
            ModalitiesInStudy='*CT*')
        studies_DX  = queryStudies(orthanc, PACS, StudyDate=date_ranges,
            ModalitiesInStudy='*DX*')
        all_studies = [studies_CT, studies_DX]
        studies_df = studieslist2df[all_studies]
    '''
    studiesList = list(itertools.chain(*studiesList))
    studiesList = [x for x in studiesList if x != []] # Remove empty elements
    studiesList_DF = pd.DataFrame(studiesList)
    return studiesList_DF

def serieslist2df(seriesList):
    """
    Converts the output of getSeriesfromStudyDF into a dataframe.
    Inputs:
      - seriesList [list]: List of series queries, such as that returned by
            getSeriesfromStudyDF.
    Outputs:
      - series_df [dataframe]: Dataframe containining all series
    Usage:
        series_list = getSeriesfromStudyDF(study_df)
        series_df = serieslist2df(series_list)
    """
    #seriesList is a list of lists, each individual list corresponds to one study
    return pd.DataFrame(list(itertools.chain.from_iterable(seriesList)))
    
def getDetails(orthanc,query):
    """
    Returns the details of the orthanc 
    """
    queryID = query[0]
    queryInd = query[1]
    
    try: details = orthanc.get_content_of_specified_query_answer_in_simplified_version(queryID,queryInd)
    except: details = []
        
    return details

def findStudies(orthanc,PACS,AccessionNumber="",PatientName="",PatientID="",PatientBirthDate="",PatientSex="",
                StudyDate="",StudyDescription="",ModalitiesInStudy=""):
    remote_modality = RemoteModality(orthanc, PACS)
    data = {'Level': 'Study', 'Query': {"StudyDate":StudyDate,
                                        "AccessionNumber":AccessionNumber,
                                        "PatientName":PatientName,
                                        "PatientID":PatientID,
                                        "PatientBirthDate":PatientBirthDate,
                                        "PatientSex":PatientSex,
                                        "StudyDescription":StudyDescription,
                                        "NumberOfStudyRelatedSeries":"",
                                        "ModalitiesInStudy":ModalitiesInStudy}}

    query_response = None
    attempts = 0
    while query_response == None and attempts<10:
        try:
            query_response = remote_modality.query(data=data)
        except:
            time.sleep(0.1)
            attempts+=1
        
    return query_response

def findSeries(orthanc,PACS,AccessionNumber):
    remote_modality = RemoteModality(orthanc, PACS)
    data = {'Level': 'Series', 'Query': {"AccessionNumber":AccessionNumber,
                                        "SeriesDescription":"",
                                        "NumberOfSeriesRelatedInstances":"",
                                        "Modality":""}}
    
    query_response = None
    attempts = 0
    while query_response == None and attempts<10:
        try:
            query_response = remote_modality.query(data=data)
        except:
            time.sleep(0.1)
            attempts+=1
        
    return query_response

# Query structure: Study Description, Modality, dateRanges
# def queryStudiesModality(orthanc,PACS,studyDesc,modality,dateRanges):
def queryStudies(
    orthanc,PACS,AccessionNumber="",PatientName="",PatientID="",
    PatientBirthDate="",PatientSex="", StudyDate="", StudyDescription="",
    ModalitiesInStudy=""
    ):
    """
    Inputs
    """
    if StudyDate == "": StudyDate = ["20000101-20291231"]
        
    studiesDetails = []
    for dateRange in StudyDate:
        print("Searching date range: {}".format(dateRange)) 
        findStudiesResponse = None
        retries = 0
        while findStudiesResponse is None and retries < 20:
            try:
                findStudiesResponse = findStudies(
                    orthanc,
                    PACS,
                    AccessionNumber=AccessionNumber,
                    PatientName=PatientName,
                    PatientID=PatientID,
                    PatientBirthDate=PatientBirthDate,
                    PatientSex=PatientSex,
                    StudyDate=dateRange,
                    StudyDescription=StudyDescription,
                    ModalitiesInStudy=ModalitiesInStudy
                )
                fStudies = orthanc.get_query_answers(findStudiesResponse['ID'])
                nStudies = len(fStudies)
                print(len(fStudies),'studies found! Retrieving Study details...')

                with ThreadPoolExecutor(max_workers=16) as pool:
                    studyDetails = list(tqdm(
                        pool.map(getDetails,repeat(orthanc),[(findStudiesResponse['ID'],studyInd) for studyInd in fStudies]),total=nStudies)
                    )

                studiesDetails.append(studyDetails)
            except: 
                print(
                    'Error in queryStudies search! Attempt {}'.format(retries)
                )
                retries = retries+1
        
        if retries >= 20:
            raise ConnectionError

    studiesDetails = list(itertools.chain(*studiesDetails))
    
    return studiesDetails

def getSeriesfromStudyDF(orthanc,PACS,studiesDF):
    nStudies=len(studiesDF)
    print(nStudies,'studies found! Retrieving Series details...')
    seriesDetails = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            seriesDetails = list(tqdm(
                pool.map(getSeriesfromStudyAccNum,repeat(orthanc),repeat(PACS),
                                          list(studiesDF['AccessionNumber'])),total=nStudies)
            )
    except: 
        print('Error in getSeriesfromStudyAccNum search!')
        raise ConnectionError

    return seriesDetails

# def queryStudiesSeries(orthanc,PACS,studiesDF):
def getSeriesfromStudyAccNum(orthanc,PACS,accNumber):
    try:
        findSeriesResponse = findSeries(orthanc,PACS,accNumber)
        fSeries = orthanc.get_query_answers(findSeriesResponse['ID'])
        seriesDetails = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            seriesDetails = list(pool.map(getDetails,repeat(orthanc),
                                          [(findSeriesResponse['ID'],seriesInd) for seriesInd in fSeries]))
    except:
        print('Error in getSeriesfromStudyAccNum function!')
        seriesDetails = []
    return seriesDetails

def genDeltaStudyDate(study,deltamonths=6):
    #find dates 6 months pre/post
    study_date_1 = (datetime.strptime(str(study['StudyDate']),'%Y%m%d') - relativedelta(months=deltamonths)).strftime("%Y%m%d")
    study_date_2 = (datetime.strptime(str(study['StudyDate']),'%Y%m%d') + relativedelta(months=deltamonths)).strftime("%Y%m%d")
    studyDateRange = str(study_date_1)+'-'+str(study_date_2)   
    return studyDateRange

def queryLinkedStudies(studiesDF,orthanc,PACS,deltamonths=6,
                       AccessionNumber="",PatientName="",PatientID="",PatientBirthDate="",PatientSex="",
                       StudyDate="",StudyDescription="",ModalitiesInStudy=""):
    
    linkedStudies = pd.DataFrame()
    linkedStudies['Group'] = ''

    for index, study in tqdm(studiesDF.iterrows(),total=len(studiesDF)):
        # get patient details
        PatientID = study['PatientID']
        PatientBirthDate = study['PatientBirthDate']
        PatientSex = study['PatientSex']    

        studyDateRange = [genDeltaStudyDate(study,deltamonths)]

        findStudiesResponse = queryStudies(orthanc,PACS,
                                           AccessionNumber=AccessionNumber,PatientName=PatientName,
                                           PatientID=PatientID,PatientBirthDate=PatientBirthDate,PatientSex=PatientSex,
                                           StudyDescription=StudyDescription,StudyDate=studyDateRange,ModalitiesInStudy=ModalitiesInStudy)
        
        if len(findStudiesResponse)>0:
            study_T = pd.DataFrame(study).T
            study_T['Group'] = index

            linkedStudy = studieslist2df(findStudiesResponse)
            linkedStudy['Group'] = index
        
            linkedStudies = pd.concat([linkedStudies,
                                     study_T,
                                     linkedStudy])
    return linkedStudies

def queryPatientStudies(patientsDF,orthanc,PACS,fields):

    patientsStudies = pd.DataFrame()
    
    for index, patient in tqdm(patientsDF.iterrows(),total=len(patientsDF)):
        
        dicomStudyFields = {
            'AccessionNumber':'',
            'PatientID':'',
            'PatientBirthDate':'',
            'PatientSex':'',
            'StudyDescription':'',
            'StudyDate':'',
            'ModalitiesInStudy':''
        }
        
        for field in fields: dicomStudyFields[field] = patient[field]
        
        AccessionNumber = dicomStudyFields['AccessionNumber']
        PatientID = dicomStudyFields['PatientID']
        PatientBirthDate = dicomStudyFields['PatientBirthDate']
        PatientSex = dicomStudyFields['PatientSex']
        StudyDescription = dicomStudyFields['StudyDescription']
        StudyDate = dicomStudyFields['StudyDate'] #['20000101-20211231'] #dicomStudyFields['StudyDate']
        ModalitiesInStudy = dicomStudyFields['ModalitiesInStudy']
        
        print(dicomStudyFields)
        findStudiesResponse = queryStudies(orthanc,PACS,
                                           AccessionNumber=AccessionNumber,
                                           PatientID=PatientID,PatientBirthDate=PatientBirthDate,PatientSex=PatientSex,
                                           StudyDescription=StudyDescription,StudyDate=StudyDate,ModalitiesInStudy=ModalitiesInStudy)        

        if len(findStudiesResponse)>0:
            patientStudies = studieslist2df(findStudiesResponse)
            patientStudies['index'] = index
            patientsStudies = pd.concat([patientsStudies,
                                     patientStudies])
        
    patientsStudies.set_index('index')
#     patientsStudies.reset_index(inplace=True,drop=True)
    
    return patientsStudies

#####################################################################
#####################################################################
# auxiliary functions for extraction

def months_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return round(abs((d2 - d1).days)/(365.25 / 12))

def save_zip(z,targetdir):
    file_like_object = io.BytesIO(z)
    with ZipFile(file_like_object,"r") as zip_ref:
        zip_ref.extractall(targetdir)
    file_like_object.close()
    
# def retrieveStudyByIUID(studyIUID,orthanc,PACS):
#     print(studyIUID)
#     remote_modality = RemoteModality(orthanc,PACS)
#     data = {"Level":"Study","Query":{"StudyInstanceUID":studyIUID}}
#     query_response = remote_modality.query(data=data)
#     retrieve_response = None
#     retrieve_attempts = 0
#     while retrieve_response is None and retrieve_attempts<10:
#         retrieve_response = remote_modality.move(query_response['ID'], {'TargetAet': TargetAET, "Synchronous": True})
#         retrieve_attempts+=1
#     return retrieve_response

def retrieveStudyByAccession(accessionNumber,orthanc,PACS,TargetAET,max_attempts=3):
    print(accessionNumber)
    remote_modality = RemoteModality(orthanc, PACS)
    data = {"Level":"Study","Query":{"AccessionNumber":accessionNumber}}
    query_response = remote_modality.query(data=data)
    retrieve_response = None
    retrieve_attempts = 0
    while retrieve_response is None and retrieve_attempts<max_attempts:
        try:
            retrieve_response = remote_modality.move(query_response['ID'], {'TargetAet': TargetAET, "Synchronous": True})
        except:
            time.sleep(0.1)
            retrieve_attempts+=1
    return retrieve_response

def retrieveByIUID(level,IUID,orthanc,PACS,TargetAET,max_attempts=3):
#     print(level,IUID)
    remote_modality = RemoteModality(orthanc, PACS)
    if level == "study": data = {"Level":"Study","Query":{"StudyInstanceUID":IUID}}
    elif level == "series": data = {"Level":"Series","Query":{"SeriesInstanceUID":IUID}}
    query_response = remote_modality.query(data=data)
    retrieve_response = None
    retrieve_attempts = 0
    while retrieve_response is None and retrieve_attempts<max_attempts:
        try:
            retrieve_response = remote_modality.move(query_response['ID'], {'TargetAet': TargetAET, "Synchronous": True})
        except:
            time.sleep(0.1)
            retrieve_attempts+=1
    return retrieve_response

def clearOrthanc(orthanc):
    allPatients = orthanc.get_patients()
    for p in allPatients:
        orthanc.delete_patient(p)
    return 0

def clearStudy(studyID,orthanc):
    delete_response = None
    try: delete_response = orthanc.delete_study(studyID)
    except: print('Unable to delete!')
    return delete_response

def genRandomID(): return str(uuid.uuid4())

def addID(sInfo,Issuer,ID):
    if Issuer in sInfo['PatientIDs']: 
        if ID not in sInfo['PatientIDs'][Issuer]: sInfo['PatientIDs'][IssuerPatID] = ID
    else: sInfo['PatientIDs'][Issuer] = ID    
    return sInfo

def getSeriesInfo(series,parent,orthanc):
    
    seriesInfo = {}
    seriesDCMInfo = orthanc.get_series_information(series)
    
    seriesInfo['ParentStudy'] = parent
    seriesInfo['SeriesID'] = genRandomID()
    
    tags = ['SeriesDescription','SeriesInstanceUID','Modality','Manufacturer']

    for tag in tags: 
        if tag in seriesDCMInfo['MainDicomTags']: seriesInfo[tag] = seriesDCMInfo['MainDicomTags'][tag]
    seriesInfo['SeriesInstances'] = len(seriesDCMInfo['Instances'])

    try: 
#         print('before',seriesInfo['SeriesDescription'])
#         seriesInfo['SeriesDescription'] = seriesInfo['SeriesDescription'].apply(lambda x: str(x).replace("/", " ").replace("\\", " "))
        seriesInfo['SeriesDescription'] = str(seriesInfo['SeriesDescription']).replace("  "," ").replace("/"," ").replace("\\"," ")
#         print('after',seriesInfo['SeriesDescription'])
    except: 
        print('Error removing / \\ from',seriesInfo['SeriesDescription'])
        
    return seriesInfo

def getStudySeriesInfo(studyID,orthanc):
    # For data extraction, not filtering
    studyInfo = {}
    studyInfo['StudyID'] = genRandomID() 

    studyDCMInfo = orthanc.get_study_information(studyID)
    
    try:
        studyInfo['StudyDescription'] = studyDCMInfo['MainDicomTags']['StudyDescription']
        studyInfo['StudyInstanceUID'] = studyDCMInfo['MainDicomTags']['StudyInstanceUID']
        studyInfo['AccessionNumber'] = studyDCMInfo['MainDicomTags']['AccessionNumber'] 
        studyInfo['InstitutionName'] = studyDCMInfo['MainDicomTags']['InstitutionName'] 
        studyInfo['StudyDate'] = studyDCMInfo['MainDicomTags']['StudyDate']
        studyInfo['PatientIDs'] = {}
        studyInfo['PatientSex'] = studyDCMInfo['PatientMainDicomTags']['PatientSex']
        studyInfo['PatientAgeMonths'] = months_between(studyDCMInfo['MainDicomTags']['StudyDate'],studyDCMInfo['PatientMainDicomTags']['PatientBirthDate'])

        studyInfo['Modalities'] = []

        studyInfo['StudySeries'] = len(studyDCMInfo['Series'])
        studyInfo['Series'] = {}
    except:
        print('Error assigning studyInfo')
    
    #Iterate through series
    for seriesInd,series in enumerate(studyDCMInfo['Series']):
        try:
            seriesInfo = getSeriesInfo(series,studyInfo['StudyID'],orthanc)
            studyInfo['Series'][seriesInd] = seriesInfo
        except:
            print('Error iterating getSeriesInfo!')
        
    return studyInfo

def procOutputName(inputName):
    return inputName.replace(" ", "_").lower() 

def checkPathExists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def clearAccessionNumber(accNumber,orthanc):
    delete_success = True
    orthancStudies = orthanc.get_studies()
    
    for oStudy in orthancStudies:
        if accNumber == orthanc.get_study_information(oStudy)['MainDicomTags']['AccessionNumber']:
            orthancStudyID = orthanc.get_study_information(oStudy)['ID']

            try: delete_response = orthanc.delete_study(orthancStudyID)
            except: 
                delete_success = False
                time.sleep(0.1)
                print('Unable to delete!')
    return delete_success

def mergeStudies(accNumber,orthanc,max_attempts=10):
    firstOrthancStudyID = ''    
    merge_attempts=0
    
    while firstOrthancStudyID=='' and merge_attempts<10:
        time.sleep(0.1)
        mergeStudyList=[]
        orthancStudies = orthanc.get_studies()
        #match 
        try:
            for oStudy in orthancStudies:
                if accNumber == orthanc.get_study_information(oStudy)['MainDicomTags']['AccessionNumber']:
                    orthancStudyID = orthanc.get_study_information(oStudy)['ID']
                    mergeStudyList.append(orthancStudyID)

        except: print('error with merge list',merge_attempts)

        #now merge
        if len(mergeStudyList)>0:
            firstOrthancStudyID = mergeStudyList[0]
#             orthancStudiesIDDict[accNumber] = firstOrthancStudyID

            if len(mergeStudyList)>1:
                try: orthanc.merge_study(firstOrthancStudyID,{"Resources":mergeStudyList[1:]})
                except: print('error with merge!')
        merge_attempts+=1
        
    return firstOrthancStudyID
        
def dicom_dataset_to_dict(dicom_header):
    dicom_dict = {}
    repr(dicom_header)
    for dicom_value in dicom_header.values():        
        if dicom_value.is_private: continue
        if dicom_value.tag.group == 16: continue #skip patient identifiers
        if dicom_value.tag == (0x7fe0, 0x0010): continue # discard pixel data
        if type(dicom_value.value) == pydicom.dataset.Dataset:
            dicomTagDesc = dicom_value.description().replace(' ','').replace('(','').replace(')','')
            dicom_dict[dicomTagDesc] = dicom_dataset_to_dict(dicom_value.value)
        else:
            v = _convert_value(dicom_value.value)
            dicomTagDesc = dicom_value.description().replace(' ','').replace('(','').replace(')','').replace("'s",'')
            dicom_dict[dicomTagDesc] = v
    return dicom_dict

def _sanitise_unicode(s):
    return s.replace(u"\u0000", "").strip()

def _convert_value(v):
    t = type(v)
    if t in (list, int, float):
        cv = v
    elif t == str:
        cv = _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode('ascii', 'replace')
        cv = _sanitise_unicode(s)
    elif t == pydicom.valuerep.DSfloat:
        cv = float(v)
    elif t == pydicom.valuerep.IS:
        cv = int(v)
    elif t == pydicom.valuerep.PersonName:
        cv = str(v)
    else:
        cv = repr(v)
    return cv



#####################################################################
#####################################################################
# main extraction loop

def extractDCM(accNumberList,orthanc,PACS,TargetAET,dcm2niixPath,outputRoot,outputZip):
    
    ##################### Step 1. DICOM get study #####################
#     retrieveByIUID('study',studyIUID)
    
    if len(accNumberList) == 3: #selected seriesIUID passed, same accNumber
        #e.g.         [[accNumb, accNumb, accNumb],[level,level,level],[IUID,IUID,IUID]]
        accNumber=accNumberList[0][0]
        print(accNumber, 'Number of series',len(accNumberList[0]))
        for ind,iuid in enumerate(accNumberList[2]):
            level=accNumberList[1][ind]
#             iuid=accNumberList[2][ind]
            try: retrieveByIUID(level,iuid,orthanc,PACS,TargetAET)
            except: print('error in retrieveByIUID!')
    else: #entire study/accession number passed
        accNumber = accNumberList
        try: retrieveStudyByAccession(accNumber,orthanc,PACS,TargetAET)
        except: print('error in retrieveStudyByAccession!')
        

#         retrieveStudyByAccession(accNumber,orthanc,PACS,TargetAET)

        
        #retrieveByIUID(level,IUID,orthanc,PACS,TargetAET):

    ##################### Step 1a. Match OrthancID to StudyIUID/accNumber +/- Merge accession numbers #####################
    try:
        orthancStudyID = mergeStudies(accNumber,orthanc)
        if orthancStudyID == '': 
            clearAccessionNumber(accNumber,orthanc)
            return
    except:
        clearAccessionNumber(accNumber,orthanc)
        print('failed merge')
        
    try:
        ##################### Step 2. Get study data #####################
        try:
            studyInfo = getStudySeriesInfo(orthancStudyID,orthanc) #creates random IDs for new studies and series
        except:
            print('Error getting Study-Series info')
            
        ##################### Step 3. Define output paths #####################
        outputPathXR = os.path.join(outputRoot,'XR',studyInfo['StudyID']) 
        outputPathCT = os.path.join(outputRoot,'CT',studyInfo['StudyID'])
        outputPathMR = os.path.join(outputRoot,'MR',studyInfo['StudyID']) 
        outputPathSR = os.path.join(outputRoot,'SR') 
        outputPathJSON = os.path.join(outputRoot,'JSON') 

        checkPathExists(outputPathJSON) #creates output path if not exist

        outputZipPath = os.path.join(outputZip,studyInfo['StudyID']) 

        ##################### Step 4. Save and extract Zip #####################
        z = orthanc.get_study_zip_file(orthancStudyID)
        save_zip(z,outputZipPath)

        ##################### Step 5. Convert DICOM format #####################

        #Save each study
        seriesList = glob.glob(outputZipPath +'\\*\\*\\*')    
        outputFileList = []

        for series in seriesList:

            instanceList = glob.glob(series + '\\*')
            seriesLen = len(instanceList)

            #grab data from first instance
            dataset = dcmread(instanceList[0])

            #extract patient IDs
            try: 
                IssuerPatID = dataset[(0x0010,0x0021)][:]
                PatID = dataset[(0x0010,0x0020)][:]
                studyInfo = addID(studyInfo,IssuerPatID,PatID)
            except: print('Error extracting Patient IDs')
                
            try: 
#                 nID = len(dataset[(0x0010,0x1002)][:])
                for dfields in dataset[(0x0010,0x1002)][:]:
                    try:
                        IssuerPatID = dfields[(0x0010,0x0021)][:]
                        PatID = dfields[(0x0010,0x0020)][:]
                        studyInfo = addID(studyInfo,IssuerPatID,PatID)
                    except: 0
            except: 
                print('Error adding PatientIDs')

            try:
                #match SeriesInstanceUID
                seriesInd = [s for s in studyInfo['Series'] if studyInfo['Series'][s]['SeriesInstanceUID'] == dataset.SeriesInstanceUID][0]

                #Add DICOM tags to studyInfo dict
                seriesModality = dataset[(0x0008, 0x0060)][:]
                if seriesModality not in studyInfo['Modalities']: studyInfo['Modalities'].append(seriesModality)
            except: print('Error matching SeriesIUID')
                
            try:
                if 'DX' in studyInfo['Modalities'] or 'CR' in studyInfo['Modalities'] or 'OT' in studyInfo['Modalities']: checkPathExists(outputPathXR) #creates output path if not exist
                if 'CT' in studyInfo['Modalities']: checkPathExists(outputPathCT) #creates output path if not exist
                if 'MR' in studyInfo['Modalities']: checkPathExists(outputPathMR) #creates output path if not exist
                if 'SR' in studyInfo['Modalities']: checkPathExists(outputPathSR) #creates output path if not exist
            except: print('Error creating modality path')

            #
            try: studyInfo['Series'][seriesInd]['dicomTags'] = dicom_dataset_to_dict(dataset)
            except: print('Error converting DICOM dataset to Dict')
                
            try:
                if seriesModality == 'MR' or seriesModality == 'CT':
                    try:
                        if seriesModality == 'CT': outputPathCTMR = outputPathCT
                        else:  outputPathCTMR = outputPathMR

                        studyInfo['Series'][seriesInd]['files'] = []

                        outputFile = studyInfo['Series'][seriesInd]['SeriesID'] + ' ' + studyInfo['Series'][seriesInd]['SeriesDescription'] 
                        cmd = '"%s"' %(dcm2niixPath)
                        cmd += ' -o "%s"' %(outputPathCTMR)
                        cmd += ' -f "%s"' %(outputFile) #output file name  
                        cmd += ' -z y ' #zip
                        cmd += '"%s"' %(series)

                        returned_value=subprocess.call(cmd, shell=True)

                        outputFile_nifty = outputFile + '.nii.gz'
                        outputFile_niftyJson = outputFile + '.json'

                        studyInfo['Series'][seriesInd]['files'].append(outputFile_nifty)
                    except: print('Error converting CT/MR')
                        
                if seriesModality in ['DX','CR','OT']:
                    try:
                        studyInfo['Series'][seriesInd]['files'] = []

                        if 'protocol' in studyInfo['Series'][seriesInd]['SeriesDescription'].lower(): 0
                        else:
                            #get pixel data
                            for ind,instance in enumerate(instanceList):
                                try:
                                    #extract pixel array
                                    ds = dcmread(instance)
                                    img = ds.pixel_array # get image array

                                    outputFile = studyInfo['Series'][seriesInd]['SeriesID'] + ' ' + studyInfo['Series'][seriesInd]['SeriesDescription']
                                    outputFile = procOutputName(outputFile) #Modify output name

                                    if seriesLen>1: outputFile_png = outputFile + '_' + str(10000+ind) + '.png'
                                    elif seriesLen==1: outputFile_png = outputFile + '.png'
                                except: print('Error extracting pixel data')
                                    
                                try:
                                    #convert pixel array to 8 bit
#                                     bits = int(studyInfo['Series'][seriesInd]['BitsStored'])
                                    bits = studyInfo['Series'][seriesInd]['dicomTags']['BitsStored']
                                    if bits > 8:
                                        img = apply_voi_lut(ds.pixel_array,ds)
                                        if studyInfo['Series'][seriesInd]['PhotometricInterpretation'] == "'MONOCHROME1'":
                                            img = (2**bits - 1) - img
                                        img = img/(2**bits - 1.0)*255
                                        img = img.astype(np.uint8)
                                        cv2.imwrite(os.path.join(outputPathXR,outputFile_png),img)
                                        #append to JSON structure
                                        studyInfo['Series'][seriesInd]['files'].append(outputFile_png)
                                        #append to list of files to move
                                        outputFileList.append(outputFile_png)

                                except: print('Error converting to 8 bit')
                    except: print('Error saving DX/CR/OT')
                        
                if seriesModality == 'SR':
                    try: 
                        studyInfo['Series'][seriesInd]['files'] = []
                        text = ''
                        for instance in instanceList:
                            dcmSR = dcmread(instance)
                            text += dcmSR[(0x0040, 0xa730)][3][(0x0040, 0xa730)][0][(0x0040,0xa160)][:].replace('<BR>','').replace('\n','')
                            text += '\n \n'

                        outputFile_txt = studyInfo['Series'][seriesInd]['SeriesID'] + '.txt'

                        try:
                            text_file = open(os.path.join(outputPathSR,outputFile_txt), "w")
                            n = text_file.write(text)
                            text_file.close()
                        except UnicodeEncodeError:
                            text_file = open(os.path.join(outputPathSR,outputFile_txt), "w")
                            n = text_file.write(str(text.encode('ascii', 'ignore')))
                            text_file.close()

                        #append to JSON structure
                        studyInfo['Series'][seriesInd]['files'].append(outputFile_txt)
                        #append to list of files to move
                        outputFileList.append(outputFile_txt)
                    except: print('Error saving SR',accNumber)
            except: print('Error in extract loop',accNumber)

        studyJSON = studyInfo['StudyID']+'.json'
        with open(os.path.join(outputPathJSON,studyJSON), 'w') as fp:
            json.dump(studyInfo, fp, indent=2)   
    except:
        print('Error in main loop')
    
    ##################### Step 6. Delete study from orthanc #####################
    try: clearStudy(orthancStudyID,orthanc)
    except: print('Error clearing study')
    
    try: clearAccessionNumber(accNumber,orthanc)
    except: print('Error clearing accession number')
    
    ##################### Step 7. Remove files and directory #####################
    try:
        pathsRemove = glob.glob(outputZipPath)
        for rmPath in pathsRemove:
            shutil.rmtree(rmPath)
    except: print('Unable to delete!')
        
#     return studyInfo
    return
