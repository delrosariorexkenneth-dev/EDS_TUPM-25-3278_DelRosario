import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class EngineeringPipeline:
    def __init__(self, data_path: str):
        """
        Initializes the pipeline architecture with the path to the original dataset.
        """
        self.raw_data_path = data_path
        self.df = None  # Holds your mathematically unique data slice

    def ingest_data(self):
        """
        Step 1: Data Ingestion & Unique Programmatic Filtering
        """
        print("="*50)
        print("Executing Step 1: Data Ingestion & Filtering")
        print("="*50)
        try:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Target file not found at '{self.raw_data_path}'")
            full_df = pd.read_csv(self.raw_data_path)
            print(f"[SUCCESS] Raw dataset loaded. Master Row Count: {len(full_df)}")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to read file: {e}")
            return False

        try:
            print("\nApplying custom programmatic filter rule...")
            self.df = full_df.iloc[::2].copy() 
            print(f"[SUCCESS] Unique analytical subset isolated.")
            print(f"Your unique dataset row count is now: {len(self.df)}\n")
            return True
        except Exception as e:
            print(f"[ERROR] Failed while applying filter logic: {e}")
            return False

    def clean_data(self):
        """
        Step 2: Automated Data Cleaning
        """
        print("="*50)
        print("Executing Step 2: Automated Data Cleaning")
        print("="*50)
        if self.df is None: return False
        try:
            self.df = self.df.drop_duplicates()
            null_counts = self.df.isnull().sum().sum()
            if null_counts > 0:
                self.df = self.df.fillna(self.df.bfill())
            
            cleaned_path = os.path.join("data", "dataset_cleaned.csv")
            self.df.to_csv(cleaned_path, index=False)
            print(f"[SUCCESS] Cleaned data exported to '{cleaned_path}'.\n")
            return True
        except Exception as e:
            print(f"[CRITICAL ERROR] Cleaning stage failed: {e}")
            return False

    def analyze_data(self):
        """
        Step 3: Engineering Data Analytics (NumPy Driven)
        """
        print("="*50)
        print("Executing Step 3: Engineering Data Analytics")
        print("="*50)
        if self.df is None: return False
        try:
            self.latency_col = 'inference_time_ms'
            if self.latency_col not in self.df.columns:
                self.latency_col = [col for col in self.df.columns if 'time' in col or 'latency' in col][0]
                
            latency_array = self.df[self.latency_col].to_numpy()
            
            mean_val = np.mean(latency_array)
            median_val = np.median(latency_array)
            std_dev = np.std(latency_array)
            variance_val = np.var(latency_array)
            
            print(f"  ▶ Mean Latency:       {mean_val:.2f} ms")
            print(f"  ▶ Median Latency:     {median_val:.2f} ms")
            print(f"  ▶ Standard Deviation: {std_dev:.2f} ms")
            print(f"  ▶ Variance (Spread):  {variance_val:.2f} ms^2")
            
            q75, q25 = np.percentile(latency_array, [75 ,25])
            iqr = q75 - q25
            self.upper_bound = q75 + (1.5 * iqr)
            
            spikes = latency_array[latency_array > self.upper_bound]
            print(f"\n[SYSTEM ALERT] Outlier Detection Profile:")
            print(f"  ▶ Critical Spike Boundary: > {self.upper_bound:.2f} ms")
            print(f"  ▶ Total Inference Spikes Detected: {len(spikes)} incidents")
            print("="*50 + "\n")
            return True
        except Exception as e:
            print(f"[CRITICAL ERROR] Statistical computation failed: {e}")
            return False

    def generate_visualizations(self):
        """
        Step 4: Visualization & Animation Suite
        Generates 3 required static graphs and 2 dynamic animations.
        """
        print("="*50)
        print("Executing Step 4: Visualizations & Animations")
        print("="*50)
        
        if self.df is None:
            print("[ERROR] No working dataset found for processing visualizations.")
            return
            
        # Ensure outputs folder exists
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Pull data arrays
        latency = self.df[self.latency_col].to_numpy()
        
        # Look for tokens or parameters columns for multi-variable plotting
        token_col = [col for col in self.df.columns if 'token' in col or 'param' in col or 'size' in col]
        secondary_data = self.df[token_col[0]].to_numpy() if token_col else np.arange(len(latency))
        secondary_name = token_col[0] if token_col else "Observation Index"

        # ----------------------------------------------------
        # STATIC GRAPH 1: HISTOGRAM (Distribution Analysis)
        # ----------------------------------------------------
        plt.figure(figsize=(7, 4))
        plt.hist(latency, bins=15, color='royalblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(latency), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({np.mean(latency):.1f} ms)')
        plt.title('AIP-01: Inference Latency Profile Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'static_1_histogram.png'), dpi=200)
        plt.close()
        print("[VISUAL] Saved Static Plot 1: Latency Histogram")

        # ----------------------------------------------------
        # STATIC GRAPH 2: BOXPLOT (Outlier Detection Structure)
        # ----------------------------------------------------
        plt.figure(figsize=(5, 4))
        plt.boxplot(latency, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
        plt.title('System Latency Outlier Profile')
        plt.ylabel('Inference Latency (ms)')
        plt.xticks([1], ['Dataset Slice'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'static_2_boxplot.png'), dpi=200)
        plt.close()
        print("[VISUAL] Saved Static Plot 2: Latency Boxplot")

        # ----------------------------------------------------
        # STATIC GRAPH 3: SCATTER PLOT (Correlation Analysis)
        # ----------------------------------------------------
        plt.figure(figsize=(7, 4))
        plt.scatter(secondary_data, latency, color='darkorange', alpha=0.6, edgecolor='k')
        plt.title(f'Latency Analysis vs {secondary_name}')
        plt.xlabel(secondary_name)
        plt.ylabel('Latency (ms)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'static_3_scatterplot.png'), dpi=200)
        plt.close()
        print("[VISUAL] Saved Static Plot 3: Correlation Scatter Plot")

        # ----------------------------------------------------
        # ANIMATED GRAPH 1: REAL-TIME SYSTEM MONITOR TREND
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4))
        x_data, y_data = [], []
        line, = ax.plot([], [], 'b-', lw=2, label='Inference Step Delay')
        alert_scatter = ax.scatter([], [], color='red', s=50, zorder=5, label='Anomaly Latency Spike')
        
        ax.set_xlim(0, len(latency))
        ax.set_ylim(0, np.max(latency) * 1.1)
        ax.set_title('AIP-01: Live Inference Latency Spikes Telemetry Monitor')
        ax.set_xlabel('System Process Sequence')
        ax.set_ylabel('Latency Response Time (ms)')
        ax.axhline(self.upper_bound, color='red', linestyle=':', label='Spike Threshold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        def init_anim1():
            line.set_data([], [])
            alert_scatter.set_offsets(np.empty((0, 2)))
            return line, alert_scatter

        def update_anim1(frame):
            x_data.append(frame)
            y_data.append(latency[frame])
            line.set_data(x_data, y_data)
            
            # Find spikes up to current frame
            curr_x = np.array(x_data)
            curr_y = np.array(y_data)
            spike_mask = curr_y > self.upper_bound
            
            if np.any(spike_mask):
                alert_scatter.set_offsets(np.c_[curr_x[spike_mask], curr_y[spike_mask]])
            return line, alert_scatter

        anim1 = animation.FuncAnimation(fig, update_anim1, frames=len(latency),
                                        init_func=init_anim1, blit=True, interval=50)
        # Save animation as GIF
        anim1.save(os.path.join(output_dir, 'animated_1_telemetry.gif'), writer='pillow')
        plt.close()
        print("[VISUAL] Saved Animated Plot 1: Telemetry Monitoring Loop")

        # ----------------------------------------------------
        # ANIMATED GRAPH 2: EXPANDING DATA DISTRIBUTION DENSITY
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlim(np.min(latency) * 0.9, np.max(latency) * 1.1)
        ax.set_ylim(0, 15)  # General scale buffer for bin frequency limits
        ax.set_title('Dynamic Shift Analysis: Performance Distribution')
        ax.set_xlabel('Processing Latency Window (ms)')
        ax.set_ylabel('Observation Instance Densities')

        def update_anim2(frame):
            ax.cla() # Clear active panel frame
            ax.set_xlim(np.min(latency) * 0.9, np.max(latency) * 1.1)
            ax.set_title('Dynamic Shift Analysis: Performance Distribution')
            ax.set_xlabel('Processing Latency Window (ms)')
            ax.set_ylabel('Observation Instance Densities')
            ax.grid(True, alpha=0.3)
            
            # Draw historical distribution context dynamically stretching over frames
            slice_window = latency[:frame+5] if (frame+5) < len(latency) else latency
            ax.hist(slice_window, bins=12, color='mediumseagreen', edgecolor='black', alpha=0.7)

        anim2 = animation.FuncAnimation(fig, update_anim2, frames=range(5, len(latency), 4), interval=100)
        anim2.save(os.path.join(output_dir, 'animated_2_distribution.gif'), writer='pillow')
        plt.close()
        print("[VISUAL] Saved Animated Plot 2: Structural Density Shift Loop")
        print("\n[STATUS] All visual pipeline assets successfully rendered into tracking directories.")

if __name__ == "__main__":
    DATA_FILE = os.path.join("data", "dataset_original.csv")
    pipeline = EngineeringPipeline(data_path=DATA_FILE)
    
    if pipeline.ingest_data():
        if pipeline.clean_data():
            if pipeline.analyze_data():
                pipeline.generate_visualizations()