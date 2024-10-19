from src.python.functions import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# RUN FFT/IFFT ON DIFFERENT SIZES AND SAVE THE RESULTS
sizes = [2 ** i for i in range(2, 20)]  # Define the sizes for testing. must be power of 2
results = performTestsAndSaveResults(sizes)
results = flatten_results(results)

# Create the DataFrame
df = pd.DataFrame(results)
print(df['size'])
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
"""
# Define folder and filename with timestamp
analysis_report_folder = './results'
report_filename = f"FFT_and_IFFT_Analysis_Report_{timestamp}.pdf"

# Make plots and save to pdf
with PdfPages(analysis_report_folder + report_filename) as pdf:
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT2 of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFT2_cycles'],label='vFFT Cycles', marker='x')
    plt.ylabel('VeeR Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Instructions Count Difference Between FFT and vFFT")
    pdf.savefig()
    plt.close() 
   
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes with big O
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFT2_cycles'], label='vFFT Cycles', marker='x')
    plt.plot(df['size'], pd.DataFrame([(i*i) for i in df['size']]), label='O(n*2)', marker='o')
    plt.plot(df['size'], pd.DataFrame([i*np.log(i) for i in df['size']]), label='O(n*logn)', marker='o')
    plt.ylabel('VeeR Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Instructions Count Difference Between FFT and vFFT With Big-O")
    pdf.savefig()
    plt.close()
    # Runtime Differnce Between npFFT, FFT and vFFT of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['npFFT_time'], label='npFFT time', marker='.')
    plt.plot(df['size'], df['nFFT_time'], label='FFT time', marker='D')
    plt.plot(df['size'], df['vFFT2_time'], label='vFFT time', marker='x')
    plt.ylabel('Run time in millisecond')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Runtime Difference Between npFFT, FFT and vFFT")
    pdf.savefig()
    plt.close()
    
    #   Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_time']/df['vFFT2_time'], label='vFFT Improvement', marker='x')
    plt.ylabel('Ratio')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT run time improvement over FFTs")
    pdf.savefig()
    plt.close()
    
    #   Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['nFFT_cycles']/df['vFFT2_cycles'], label='vFFT Improvement', marker='x')
    plt.ylabel('Ratio')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT cpu instructions improvement over FFTs")
    pdf.savefig()
    plt.close()
    exit(0)
    
    # Show value difference per size
    # for idx, row in df.iterrows():
    #     size = row['size']
    #     np_fft_result = np.array(row['npFFTresult'])
    #     fft_result = np.array(row['FFTresult'])
    #     v_fft_result = np.array(row['vFFTresult'])

    #     # Create a figure for each size
    #     plt.figure(figsize=(12, 6))

    #     # Plot the abs diff of real part of the results
    #     # plt.plot(np.abs(np_fft_result - fft_result), label='FFT Diff', color='blue', linestyle='-')
    #     # plt.plot(np.abs(np_fft_result - v_fft_result), label='vFFT  Diff', color='green', linestyle='--')
     

    #     plt.plot(np.abs(np.real(np_fft_result) - np.real(fft_result)), label='FFT Real Diff', color='blue', linestyle='-')
    #     plt.plot(np.abs(np.real(np_fft_result) - np.real(v_fft_result)), label='vFFT Real Diff', color='green', linestyle='--')
     
    # # Set the title and labels
    #     plt.title(f'vFFT/FFT  REAL diff from npFFT for Size {size}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Diff')
    #     plt.legend()

    #     # Show the plot
    #     pdf.savefig()
    #     plt.close()
    #     # Create a figure for each size
    #     plt.figure(figsize=(12, 6))
        
    #     # # Optionally plot the dif of imaginary part (uncomment the following lines if needed)
    #     plt.plot(np.abs(np.imag(np_fft_result) - np.imag(fft_result)), label='FFT Imag Diff', color='red', linestyle='-.')
    #     plt.plot(np.abs(np.imag(np_fft_result) - np.imag(v_fft_result)), label='vFFT Imag Diff', color='orange', linestyle=':')

    #     # Set the title and labels
    #     plt.title(f'vFFT/FFT IMAG diff from npFFT for Size {size}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Diff')
    #     plt.legend()

    #     # Show the plot
    #     pdf.savefig()
    #     plt.close()


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
create_spreadsheet(results, 'onlyvfft_results.xlsx')



print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)


