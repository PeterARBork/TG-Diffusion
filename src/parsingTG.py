from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parseanimalside(
    rawfolder,
    parsedfolder,
    animalsidecombo,
    pixelsize_mm,
    frameduration_h,
    num_pixels_to_average_source_over,
    source_is_SAS=True,
):
    """ Turns raw data into parsed data (see definitions).
    
    Args:
        rawfolder: path
        parsedfolder: path
        animalsidecombo: string e.g. "a1_left_"
        pixelsize_mm: float
        framduration_h: float
    Returns:
        conc_matrix, source_vector, time_vector, distance_vector
    """
    
    sas_matrix = parse_to_matrix(rawfolder, animalsidecombo + "_sas")
    source_vector = sas_matrix[:, -num_pixels_to_average_source_over:].mean(axis=1)
    
    conc_matrix = parse_to_matrix(rawfolder, animalsidecombo + "_tg")
    if not source_is_SAS:
        source_vector = conc_matrix[:, 0]
        assert source_vector.size == conc_matrix.shape[0], f"source_vector.size, conc_matrix.shape[0] = {source_vector.size}, {conc_matrix.shape[0]}"

    num_frames, num_pixels = conc_matrix.shape
    source_vector = source_vector[:num_frames]
    time_vector = np.cumsum(num_frames * [frameduration_h,])
    distance_vector = np.cumsum(num_pixels * [pixelsize_mm,])
    
    assert distance_vector.size == num_pixels, f"(distance_vector.size, num_pixels) = ({distance_vector.size}, {num_pixels})"
    assert time_vector.size == num_frames
    assert time_vector.size == source_vector.size, f"(time_vector.size, source_vector.size) = ({time_vector.size}, {source_vector.size})"
    
    savetodir = parsedfolder + animalsidecombo + "/"
    np.save(savetodir + "source_vector", source_vector)
    np.save(savetodir + "concmatrix", conc_matrix)
    np.save(savetodir + "time_vector", time_vector)
    np.save(savetodir + "distance_vector", distance_vector)
    
    plt.plot(time_vector, conc_matrix[:, 0], label="conc matrix t=0")
    plt.plot(time_vector, source_vector, label="source vector")
    plt.legend()
    plt.title(animalsidecombo)
    plt.show()
    
    return conc_matrix, source_vector, time_vector, distance_vector

def switch_left_sastg(f):
    sas_or_tg = "sas" if "sas" in f else "tg"
    left_or_right = "left" if "left" in f else "right"
    
    sas_or_tg_last = True if f[-2:] == "as" or f[-2:] == "tg" else False
    
    if sas_or_tg_last:
        return f[:3] + sas_or_tg + "_" + left_or_right
    else:
        return f[:3] + left_or_right + "_" + sas_or_tg
    
def parse_to_matrix(
    rawfolder,
    basename,
):
    csv_files = [f for f in listdir(rawfolder)
                 if basename in f and ".csv" in f]
    if len(csv_files) == 0:
        csv_files = [f for f in listdir(rawfolder)
                     if basename in switch_left_sastg(f) and ".csv" in f]
        
        if len(csv_files) == 0:
            print(f"no CSV files found for {basename}")
            return np.nan * np.zeros((2, 2))
    
    def extract_framenum(filename):
        try:
            return int(filename[len(basename):].replace('.csv', ''))
        except ValueError as ve:
            print(filename)
            raise ve
        
    sorted_csv_files = sorted(csv_files, key=extract_framenum)

    raw_df_list = []
    for filename in sorted_csv_files:
        df = pd.read_csv(rawfolder + filename, )
        raw_df_list.append(df['Results'])

    conc_matrix_df = pd.concat(raw_df_list, axis=1, ignore_index=True)
    conc_matrix = np.transpose(conc_matrix_df.values)
    
    return conc_matrix

