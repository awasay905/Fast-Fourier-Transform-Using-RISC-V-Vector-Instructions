from src.python.functions import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import seaborn as sns

# First RUN FFT/IFFT ON DIFFERENT SIZES AND SAVE THE RESULTS
# Define the sizes for testing. must be power of 2. You can change this to run on different sizes
sizes = [2 ** i for i in range(2, 16)]  
results = performTestsAndSaveResults(sizes)
results = flatten_results(results)
fig_size = (12,7)
# Create a pandas DataFrame from the results
df = pd.DataFrame(results)

# Create a timestamp for naming the results PDF
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the folder and filename with timestamp
results_folder = 'results/analysis/'
report_filename = f'fft_and_ifft_analysis_report_{timestamp}.pdf'

"""
The data has array of size and input, 
and then result, cycles, time for
npFFT/npIFFT
nFFT/nIFFT
nFFT2/nIFFT
vFFT/vIFFT
vFFT2/vIFFT2
"""


# Make plots and save to pdf
with PdfPages(results_folder + report_filename) as pdf:
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT2 of Different input sizes
    plt.figure(figsize=fig_size)
    plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFT2_cycles'],label='vFFT Cycles', marker='x')
    plt.ylabel('Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("VeeR Instruction Count for FFT and vFFT")
    pdf.savefig()
    plt.close() 
    
    # # Improvement of vFFT over FFT of Different input sizes i.e ratio
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['size'], df['nFFT_cycles']/df['vFFT2_cycles'], label='vFFT Improvement', marker='x')
    # plt.ylabel('Improvement')
    # plt.xlabel('Input Size (log)')
    # plt.xscale('log')
    # plt.legend()
    # plt.title("vFFT VeeR instructions count improvement over FFT")
    # pdf.savefig()
    # plt.close()
   
    # # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes with big O
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['size'], df['nFFT_cycles'], label='FFT Cycles', marker='D')
    # plt.plot(df['size'], df['vFFT2_cycles'], label='vFFT Cycles', marker='x')
    # plt.plot(df['size'], pd.DataFrame([(i*i) for i in df['size']]), label='O(n*2)', marker='o')
    # plt.plot(df['size'], pd.DataFrame([i*np.log(i) for i in df['size']]), label='O(n*logn)', marker='o')
    # plt.ylabel('Instruction Count')
    # plt.xlabel('Input Size (log)')
    # plt.xscale('log')
    # plt.legend()
    # plt.title("VeeR Instruction Count for FFT and vFFT With Big-O")
    # pdf.savefig()
    # plt.close()
    
    sizes = df['size']
    nFFT_times = df['nFFT_time']
    vFFT2_times = df['vFFT2_time']
    nFFT_cycles = df['nFFT_cycles']
    vFFT2_cycles = df['vFFT2_cycles']

    # Speedup calculation (time and CPU cycles)
    speedup_time = nFFT_times / vFFT2_times
    speedup_cycles = nFFT_cycles / vFFT2_cycles
    
        # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('CPU Cycles and Runtime Comparision of nFFT and vFFT', fontsize=14)
    
    # Speedup Plot (Time)
    axs[1, 1].plot(sizes, speedup_time, label='Speedup (Time)', marker='o', color='blue')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Input Size (log scale)')
    axs[1, 1].set_ylabel('Speedup (nFFT time / vFFT2 time)')
    axs[1, 1].set_title('Runtime Speedup')
    axs[1, 1].legend()
    
    # Speedup Plot (Cycles)
    axs[0, 1].plot(sizes, speedup_cycles, label='Speedup (Cycles)', marker='x', color='red')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Input Size (log scale)')
    axs[0, 1].set_ylabel('Speedup (nFFT cycles / vFFT2 cycles)')
    axs[0, 1].set_title('VeeR Cycvle Speedup')
    axs[0, 1].legend()
    
    # Bar Plot (Time)
    width = 0.35  # Width of the bars
    x = np.arange(len(sizes))  # Label locations
    
    axs[1, 0].bar(x - width/2, nFFT_times, width, label='nFFT Time', color='blue')
    axs[1, 0].bar(x + width/2, vFFT2_times, width, label='vFFT2 Time', color='orange')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(sizes, rotation=45)
    axs[1, 0].set_xlabel('Input Size')
    axs[1, 0].set_ylabel('Time (ms)')
    axs[1, 0].set_title('Runtime Time Comparison')
    axs[1, 0].legend()

    # Bar Plot (Cycles)
    axs[0, 0].bar(x - width/2, nFFT_cycles, width, label='nFFT Cycles', color='blue')
    axs[0, 0].bar(x + width/2, vFFT2_cycles, width, label='vFFT2 Cycles', color='orange')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(sizes, rotation=45)
    axs[0, 0].set_xlabel('Input Size')
    axs[0, 0].set_ylabel('CPU Cycles')
    axs[0, 0].set_title('VeeR Cycles Comparison')
    axs[0, 0].legend()

    # Adjust layout and save the entire page to the PDF
    plt.tight_layout()
    pdf.savefig()  # Save to PDF

    
    # NOW WE ARE DONE FOR RUN TIME. TIME FOR ERRORS
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
        nFFT_errors = size_df.apply(lambda row: np.abs(np.array(row['npFFT_result']) - np.array(row['nFFT_result'])), axis=1)
        nIFFT_errors = size_df.apply(lambda row: np.abs(np.array(row['input']) - np.array(row['nIFFT_result'])), axis=1)
        
        # Flatten the errors (in case each row has arrays of errors)
        vFFT_errors_flat = np.concatenate(vFFT_errors.values)
        vIFFT_errors_flat = np.concatenate(vIFFT_errors.values)
        nFFT_errors_flat = np.concatenate(nFFT_errors.values)
        nIFFT_errors_flat = np.concatenate(nIFFT_errors.values)
        
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
        
        # Append statistics for nFFT errors
        avg_nFFT_error.append(nFFT_errors_flat.mean())
        min_nFFT_error.append(nFFT_errors_flat.min())
        max_nFFT_error.append(nFFT_errors_flat.max())
        std_nFFT_error.append(nFFT_errors_flat.std())
        
        # Append statistics for nIFFT errors
        avg_nIFFT_error.append(nIFFT_errors_flat.mean())
        min_nIFFT_error.append(nIFFT_errors_flat.min())
        max_nIFFT_error.append(nIFFT_errors_flat.max())
        std_nIFFT_error.append(nIFFT_errors_flat.std())
    

    fig, axs = plt.subplots(2, 2, figsize=fig_size, sharey='row')
    fig.suptitle('Error Ranges for nFFT andvFFT', fontsize=14)
    
    
    # vFFT on left
    axs[0, 0].plot(df['size'], avg_vFFT_error, label='vFFT Avg Error', marker='x')
    axs[0, 0].fill_between(df['size'], min_vFFT_error, max_vFFT_error, alpha=0.3, label='vFFT Min-Max Error')
    axs[0, 0].fill_between(df['size'], np.array(avg_vFFT_error) - np.array(std_vFFT_error), 
                           np.array(avg_vFFT_error) + np.array(std_vFFT_error), alpha=0.3, label='vFFT Avg ± Std')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("vFFT")
    axs[0, 0].set_ylabel('Error')
    axs[0, 0].legend()
    
    # nFFT on right
    axs[0, 1].plot(df['size'], avg_nFFT_error, label='nFFT Avg Error', marker='x')
    axs[0, 1].fill_between(df['size'], min_nFFT_error, max_nFFT_error, alpha=0.3, label='nFFT Min-Max Error')
    axs[0, 1].fill_between(df['size'], np.array(avg_nFFT_error) - np.array(std_nFFT_error), 
                           np.array(avg_nFFT_error) + np.array(std_nFFT_error), alpha=0.3, label='nFFT Avg ± Std')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title("nFFT")
    axs[0, 1].set_ylabel('Error')
    axs[0, 1].legend()
    
    # vIFFT on bottom left
    axs[1, 0].plot(df['size'], avg_vIFFT_error, label='vIFFT Avg Error', marker='o')
    axs[1, 0].fill_between(df['size'], min_vIFFT_error, max_vIFFT_error, alpha=0.3, label='vIFFT Min-Max Error')
    axs[1, 0].fill_between(df['size'], np.array(avg_vIFFT_error) - np.array(std_vIFFT_error), 
                           np.array(avg_vIFFT_error) + np.array(std_vIFFT_error), alpha=0.3, label='vIFFT Avg ± Std')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_title("vIFFT")
    axs[1, 0].set_ylabel('Error')
    axs[1, 0].legend()
    
    # nIFFT on bottom right
    axs[1, 1].plot(df['size'], avg_nIFFT_error, label='nIFFT Avg Error', marker='o')
    axs[1, 1].fill_between(df['size'], min_nIFFT_error, max_nIFFT_error, alpha=0.3, label='nIFFT Min-Max Error')
    axs[1, 1].fill_between(df['size'], np.array(avg_nIFFT_error) - np.array(std_nIFFT_error), 
                           np.array(avg_nIFFT_error) + np.array(std_nIFFT_error), alpha=0.3, label='nIFFT Avg ± Std')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_title("nIFFT")
    axs[1, 1].set_ylabel('Error')
    axs[1, 1].legend()
    
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize=fig_size)
    plt.plot(df['size'], avg_vFFT_error, label='vFFT Avg Error', marker='x')
    plt.fill_between(df['size'], min_vFFT_error, max_vFFT_error, alpha=0.3, label='vFFT Min-Max Error')
    plt.fill_between(df['size'], np.array(avg_vFFT_error) - np.array(std_vFFT_error), 
                     np.array(avg_vFFT_error) + np.array(std_vFFT_error), alpha=0.3, label='vFFT Avg ± Std')
    plt.plot(df['size'], avg_vIFFT_error, label='vIFFT Avg Error', marker='o')
    plt.fill_between(df['size'], min_vIFFT_error, max_vIFFT_error, alpha=0.3, label='vIFFT Min-Max Error')
    plt.fill_between(df['size'], np.array(avg_vIFFT_error) - np.array(std_vIFFT_error), 
                     np.array(avg_vIFFT_error) + np.array(std_vIFFT_error), alpha=0.3, label='vIFFT Avg ± Std')
    plt.plot(df['size'], std_vFFT_error, label='vFFT Std', marker='o')
    plt.plot(df['size'], std_vIFFT_error, label='vIFFT Std', marker='o')
    plt.xscale('log')
    plt.ylabel('Error')
    plt.xlabel('Input Size (log)')
    plt.legend()
    plt.title("Avg, Min, Max, STD and Avg ± Std Deviation Plot of vFFT and vIFFT")
    pdf.savefig()
    plt.close()
    
    # Create the error statistics DataFrame for vFFT
    error_stats_vFFT = np.array([avg_vFFT_error, min_vFFT_error, max_vFFT_error, std_vFFT_error])
    error_stats_df_vFFT = pd.DataFrame(error_stats_vFFT.T, columns=['Avg', 'Min', 'Max', 'Std'], index=df['size'])

    # Create the error statistics DataFrame for nFFT
    error_stats_nFFT = np.array([avg_nFFT_error, min_nFFT_error, max_nFFT_error, std_nFFT_error])
    error_stats_df_nFFT = pd.DataFrame(error_stats_nFFT.T, columns=['Avg', 'Min', 'Max', 'Std'], index=df['size'])

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=fig_size)  # Adjust figsize as needed

    # Plot the heatmap for vFFT
    sns.heatmap(error_stats_df_vFFT, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Error Value'}, ax=axes[0])
    axes[0].set_title("vFFT")
    axes[0].set_ylabel('Input Size')

    # Plot the heatmap for nFFT
    sns.heatmap(error_stats_df_nFFT, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Error Value'}, ax=axes[1])
    axes[1].set_title("nFFT")
    axes[1].set_ylabel('Input Size')

    # Adjust layout
    fig.suptitle('Heatmap of Errors (Avg, Min, Max, Std) Across Sizes', fontsize=14)

    plt.tight_layout()

    # Save to PDF
    pdf.savefig(fig)
    plt.close(fig)

    # 7. Log-Log Plot of Errors
    plt.figure(figsize=fig_size)
    plt.loglog(df['size'], avg_vFFT_error, label='vFFT Average Error', marker='x')
    plt.loglog(df['size'], avg_vIFFT_error, label='vIFFT Average Error', marker='x')
    plt.loglog(df['size'], avg_nFFT_error, label='nFFT Average Error', marker='o')
    plt.loglog(df['size'], avg_nIFFT_error, label='nIFFT Average Error', marker='o')
    
    plt.ylabel('Error (log)')
    plt.xlabel('Input Size (log)')
    plt.legend()
    plt.title("Log-Log Plot of vFFT/nFFT and vIFFT/nFFT Errors")
    pdf.savefig()
    plt.close()
    
print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)


