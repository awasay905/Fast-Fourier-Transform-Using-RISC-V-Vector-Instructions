from src.python.functions import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# RUN FFT/IFFT ON DIFFERENT SIZES AND SAVE THE RESULTS
sizes = [2 ** i for i in range(2, 15)]  # Define the sizes for testing. must be power of 2
results = performTestsAndSaveResults(sizes)
results = flatten_results(results)



# Create the DataFrame
df = pd.DataFrame(results)
print(df.head())

# Create timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


"""
this is how data looks like
        'size': [],
        'npFFT_result': [],
        'npFFT_cycles': [],
        'npFFT_time': [],
        'npIFFT_result': [],
        'npIFFT_cycles': [],
        'npIFFT_time': [],
        'nFFT_result': [],
        'nFFT_cycles': [],
        'nFFT_time': [],
        'nIFFT_result': [],
        'nIFFT_cycles': [],
        'nIFFT_time': [],
        'nFFT2_result': [],
        'nFFT2_cycles': [],
        'nFFT2_time': [],
        'nIFFT2_result': [],
        'nIFFT2_cycles': [],
        'nIFFT2_time': [],
        'vFFT_result': [],
        'vFFT_cycles': [],
        'vFFT_time': [],
        'vIFFT_result': [],
        'vIFFT_cycles': [],
        'vIFFT_time': [],
        'vFFT2_result': [],
        'vFFT2_cycles': [],
        'vFFT2_time': [],
        'vIFFT2_result': [],
        'vIFFT2_cycles': [],
        'vIFFT2_time': []
        
        
        we are only concerned with
        npFFT, vFFT2, vIFFT2, nFFT
        npFFT is to check for error of vFFT2
        and we can compare vIFFT2 result with input to check its error
        nFFT is for time
"""
# Define folder and filename with timestamp
analysis_report_folder = './results/analysis/'
report_filename = f"FFT_and_IFFT_Analysis_Report_{timestamp}.pdf"

# Make plots and save to pdf
with PdfPages(analysis_report_folder + report_filename) as pdf:
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT2 of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFT2_cycles'],label='vFFT Cycles', marker='x')
    plt.ylabel('Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("VeeR Instruction Count for FFT and vFFT")
    pdf.savefig()
    plt.close() 
    
    # Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles']/df['vFFT2_cycles'], label='vFFT Improvement', marker='x')
    plt.ylabel('Improvement')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT VeeR instructions count improvement over FFT")
    pdf.savefig()
    plt.close()
   
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes with big O
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFT2_cycles'], label='vFFT Cycles', marker='x')
    plt.plot(df['size'], pd.DataFrame([(i*i)/50 for i in df['size']]), label='O(n*2)', marker='o')
    plt.plot(df['size'], pd.DataFrame([i*np.log(i)/50 for i in df['size']]), label='O(n*logn)', marker='o')
    plt.ylabel('Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("VeeR Instruction Count for FFT and vFFT With Big-O")
    pdf.savefig()
    plt.close()
    
    # Runtime Differnce Between npFFT, FFT and vFFT of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['npFFT_time'], label='npFFT time', marker='.')
    plt.plot(df['size'], df['nFFT_time'], label='FFT time', marker='D')
    plt.plot(df['size'], df['vFFT2_time'], label='vFFT time', marker='x')
    plt.ylabel('Run time in seconds')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Runtime of npFFT, FFT and vFFT")
    pdf.savefig()
    plt.close()
    
    #  Runtime Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_time']/df['vFFT2_time'], label='vFFT Time Improvement', marker='x')
    plt.ylabel('Ratio')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT run time improvement over FFTs")
    pdf.savefig()
    plt.close()
    
    import re



    
    # draw a graph showing avg, min, max, std error of vFFT with npFFT , and vIFFT and input for different sizes. also draw box plot 
    # basically first i need to show how much is diff in error between vFFT and nFFT
    # then i will show their diff in inverses
    avg_vFFT_error = []
    min_vFFT_error = []
    max_vFFT_error = []
    std_vFFT_error = []
    
    avg_vIFFT_error = []
    min_vIFFT_error = []
    max_vIFFT_error = []
    std_vIFFT_error = []
    
    avg_nFFT_error = []
    min_nFFT_error = []
    max_nFFT_error = []
    std_nFFT_error = []
    
    avg_nIFFT_error = []
    min_nIFFT_error = []
    max_nIFFT_error = []
    std_nIFFT_error = []
    
    for size in df['size']:
        size_df = df[df['size'] == size]
    
        # Ensure npFFT_result and vFFT_result contain comparable data
        # Apply element-wise operations using .apply()
        vFFT_errors = size_df.apply(lambda row: np.abs(np.array(row['npFFT_result']) - np.array(row['vFFT2_result'])), axis=1)
        vIFFT_errors = size_df.apply(lambda row: np.abs(np.array(row['input']) - np.array(row['vIFFT2_result'])), axis=1)
        
        # Flatten the errors (in case each row has arrays of errors)
        vFFT_errors_flat = np.concatenate(vFFT_errors.values)
        print(vFFT_errors_flat)
        vIFFT_errors_flat = np.concatenate(vIFFT_errors.values)
        
        # Append statistics for vFFT errors
        avg_vFFT_error.append(vFFT_errors_flat.mean())
        min_vFFT_error.append(vFFT_errors_flat.min())
        max_vFFT_error.append(vFFT_errors_flat.max())
        std_vFFT_error.append(vFFT_errors_flat.std())
        
        # Append statistics for vIFFT errors
        avg_vIFFT_error.append(vIFFT_errors_flat.mean())
        min_vIFFT_error.append(vIFFT_errors_flat.min())
        max_vIFFT_error.append(vIFFT_errors_flat.max())
        std_vIFFT_error.append(vIFFT_errors_flat.std())
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], avg_vFFT_error, label='vFFT Average Error', marker='x')
    plt.plot(df['size'], avg_vIFFT_error, label='vIFFT Average Error', marker='o')
    plt.fill_between(df['size'], min_vFFT_error, max_vFFT_error, alpha=0.3, label='vFFT Error Range')
    plt.fill_between(df['size'], min_vIFFT_error, max_vIFFT_error, alpha=0.3, label='vIFFT Error Range')
    plt.ylabel('Error')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Average Error of vFFT and vIFFT")
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([avg_vFFT_error, avg_vIFFT_error], labels=['vFFT Average Error', 'vIFFT Average Error'])
    plt.ylabel('Error')
    plt.title("Box Plot of vFFT and vIFFT Average Errors")
    pdf.savefig()
    plt.close()


def create_spreadsheet(results, file_name):
    import pandas as pd
    import numpy as np
    
    # Example difference function
    def diff_func(a, b):
        return abs(a - b)
    # Create an Excel writer object
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # Loop through each entry in results
        for entry in results:
            size = entry['size']
            
            data = {
                'real' : [result for result in entry['real']],
                'imag' : [result for result in entry['imag']],
                'npFFT_real': [result.real for result in entry['npFFTresult']],
                'npFFT_imag': [result.imag for result in entry['npFFTresult']],
                #'FFT_real': [result.real for result in entry['FFTresult']],
               # 'FFT_imag': [result.imag for result in entry['FFTresult']],
                'vFFT_real': [result.real for result in entry['vFFTresult']],
                'vFFT_imag': [result.imag for result in entry['vFFTresult']],
               # 'npIFFT_real': [result.real for result in entry['npIFFTresult']],
               # 'npIFFT_imag': [result.imag for result in entry['npIFFTresult']],
               # 'IFFT_real': [result.real for result in entry['IFFTresult']],
                #'IFFT_imag': [result.imag for result in entry['IFFTresult']],
                'vIFFT_real': [result.real for result in entry['vIFFTresult']],
                'vIFFT_imag': [result.imag for result in entry['vIFFTresult']],
            }
            
        
             # Calculate abs differences  between ffts
          #  data['diff_npFFT_FFT_abs'] = [diff for diff in np.abs(entry['npFFTresult'] - entry['FFTresult'])]
            data['diff_npFFT_vFFT_abs'] = [diff for diff in np.abs(entry['npFFTresult'] - entry['vFFTresult'])]
            #data['diff_FFT_vFFT_abs'] = [diff for diff in np.abs(entry['FFTresult'] - entry['vFFTresult'])]
            data['diff_npIFFT_IFFT_abs'] = [diff for diff in np.abs(entry['npIFFTresult'] - entry['IFFTresult'])]
           # data['diff_npIFFT_vIFFT_abs'] = [diff for diff in np.abs(entry['npIFFTresult'] - entry['vIFFTresult'])]
           # data['diff_IFFT_vIFFT_abs'] = [diff for diff in np.abs(entry['IFFTresult'] - entry['vIFFTresult'])]
            
       
            # Add runtime and cycles
           # data['npFFT_cycles'] = [entry['npFFTcycles']] * len(entry['npFFTresult'])
          # data['npFFT_time'] = [entry['npFFTtime']] * len(entry['npFFTresult'])
          #  data['FFT_cycles'] = [entry['FFTcycles']] * len(entry['FFTresult'])
          #  data['FFT_time'] = [entry['FFTtime']] * len(entry['FFTresult'])
          #  data['vFFT_cycles'] = [entry['vFFTcycles']] * len(entry['vFFTresult'])
           # data['vFFT_time'] = [entry['vFFTtime']] * len(entry['vFFTresult'])

            #data['npIFFT_cycles'] = [entry['npIFFTcycles']] * len(entry['npIFFTresult'])
           # data['npIFFT_time'] = [entry['npIFFTtime']] * len(entry['npIFFTresult'])
           # data['IFFT_cycles'] = [entry['IFFTcycles']] * len(entry['IFFTresult'])
           # data['IFFT_time'] = [entry['IFFTtime']] * len(entry['IFFTresult'])
          #  data['vIFFT_cycles'] = [entry['vIFFTcycles']] * len(entry['vIFFTresult'])
           # data['vIFFT_time'] = [entry['vIFFTtime']] * len(entry['vIFFTresult'])

             # in the end Calculate differences  between real and imag coutnerpart. optional
            # data['diff_npFFT_FFT_real'] = [diff_func(entry['npFFTresult'][i].real, entry['FFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_FFT_imag'] = [diff_func(entry['npFFTresult'][i].imag, entry['FFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_vFFT_real'] = [diff_func(entry['npFFTresult'][i].real, entry['vFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_vFFT_imag'] = [diff_func(entry['npFFTresult'][i].imag, entry['vFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_FFT_vFFT_real'] = [diff_func(entry['FFTresult'][i].real, entry['vFFTresult'][i].real) for i in range(len(entry['FFTresult']))]
            # data['diff_FFT_vFFT_imag'] = [diff_func(entry['FFTresult'][i].imag, entry['vFFTresult'][i].imag) for i in range(len(entry['FFTresult']))]
            # data['diff_npIFFT_IFFT_real'] = [diff_func(entry['npIFFTresult'][i].real, entry['IFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_IFFT_imag'] = [diff_func(entry['npIFFTresult'][i].imag, entry['IFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_vIFFT_real'] = [diff_func(entry['npIFFTresult'][i].real, entry['vIFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_vIFFT_imag'] = [diff_func(entry['npIFFTresult'][i].imag, entry['vIFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_IFFT_vIFFT_real'] = [diff_func(entry['IFFTresult'][i].real, entry['vIFFTresult'][i].real) for i in range(len(entry['FFTresult']))]
            # data['diff_IFFT_vIFFT_imag'] = [diff_func(entry['IFFTresult'][i].imag, entry['vIFFTresult'][i].imag) for i in range(len(entry['FFTresult']))]

            
            # Create a DataFrame for the current size
            df = pd.DataFrame(data)
            
            # Write to a new sheet named after the size
            df.to_excel(writer, sheet_name=f'Size_{size}', index=False)

# Usage
#create_spreadsheet(results, 'onlyvfft_results.xlsx')



print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)


