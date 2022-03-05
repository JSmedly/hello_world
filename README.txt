This repository creates two docker containers. One runs an Orthanc server, the other runs a Jupyter Notebook. The two containers are connected on the same network (named 'Palantir') and can listen to each other. For example, the Jupyter Notebook container can access the Orthanc server using the IP:
    'http://orthanc:8042'
Where 8042 is the http port exposed by the Orthanc container.

For the Orthanc server to run correctly, the orthanc.json configuration file needs to be correctly set. Without this file, Orthanc uses the default configuration settings. Any settings not mentioned in the orthanc.json file also use their default values. Before running the compose file, the orthanc.json needs to have its DicomModalities and RegisteredUsers specified (see the commented out examples). Make sure to remember your RegisteredUsers username and password since this needs to be used to setup credentials between the Jupyter Notebook container and the Orthanc server (see test_connections.ipynb for an example of how to do this).

For non-VS Code users:
The local ./jupter folder gets bind mounted into the Jupyter Notebook container, allowing for code to be shared and persisted locally and in the container. When opening a Jupyter Notebook from the link provided by the container, you will have to navigate through 'work/' to see the jupyter folders.
Also be aware that saving to directories outside of the jupyter file will not always have desired results. Specifically, these files may not persist outside of the containter's creation, which could result in massive data loss.

for VS Code users:
The devcontainer needs to be set to run the jupyter notebook, not the orthanc container. All files in the local repository will be copied into the container and new files or folders created in the container will be reflected on the host machine. Therefore, this method is safer and preferred.


The recommended pipeline for extracting the data is as follows.
1. Initiate contact with the orthanc server and test the database connection.
2. Follow the steps in the chest_studies_extraction.ipynb notebook. At this point it is recommended to first extract a few (~1000) studies while you are still figuring out what you want.
    2.a. Use the queryStudies function to extract studies by modality. You can also filter by study description, but it is recommended to only do this _after_ you have manually performed filtering and found the study descriptions all had a common word within them.
    2.b. Combine the extracted studies into a list, use 'studieslist2DF' to convert the list to a pandas dataframe, and remove obviously incorrect entries, like duplicates or empty slots.
    2.c. Filter studies by some high level features such as age at time of study and recorded gender. During early experiments, you can filter by studyDescription, but be broad. Instead of filtering at this point, it might be better to filter at the studies extraction point, since this discourages fine level filtering.
    2.d. Save the studies to a json file to be accessed by the series data extraction process.
3. Follow the steps in the chest_series_extraction.ipynb notebook. We will use the studies from the previous step as our starting point. During early attempts, there will only be a few studies to access, so only a few series will later be extracted. This is a good time to familiarise yourself with the different filtering options and the algorithm designed for more reliable extraction of series data. Depending on the data type, this can be a very time consuming process to get wrong.
    3.a. Connect to the orthanc server.
    3.b. Extract series data from studies using getSeriesfromStudyDF(). The checkpointer function is useful for this process since you can combat disconnect errors by checkpointing every n elements in the sequence. Then a disconnect or other exception won't cause you to loose the entirety of your data. Additionally, it is possible to sometimes force exceptions, which will immediately dump all extracted data into a .json file. If managed appropriately, this can you to restart with minimal data loss. The extraction process tends to work well checkpointed at between 1,000-10,000 elements. Any fewer and you get too many restarts, any more and you get large lag and start to lose the backup benefits of checkpointing. Additionally, the chest_series_extraction.ipynb notebook has a higher level saving system where it makes the checkpointer run on large parts of the original series. The saving system combats memory expansion and having very large data read/write times. When the checkpointer runs by itself, all the data is appeneded into one list. The saving functionality dumps this list into a json file, clearing the stack, and allowing the checkpointer to start running fresh on the next section of data. This method was found to more reliably extract data than the checkpoint in isolation.
    3.c. Upon extracting the series, the datafram is created and pre-processed.

