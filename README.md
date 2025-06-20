Of course. This is an excellent idea. Consolidating the entire journey into a single, chronological guide serves as a perfect runbook for future reference.

Here is a complete, detailed file outlining every prerequisite, command, error, and solution we worked through together to get your Mamba2 network training.

Leela Chess Zero Mamba2 Implementation: A Complete Guide

This document outlines the complete series of steps, including dependency installation and all troubleshooting, required to integrate and train a new Mamba2-based neural network architecture within the lczero-training framework.

Phase 0: System Prerequisites

This phase ensures the underlying hardware and system drivers are ready.

Verify NVIDIA GPU and Driver: The first step is to confirm the system has a recognized NVIDIA GPU and that the driver is correctly installed.

Generated bash
nvidia-smi


Success: A table appears showing the GPU name (e.g., NVIDIA RTX A4000), Driver Version, and CUDA Version.

Failure: A command not found error indicates the NVIDIA drivers must be installed at the system level first.

Phase 1: Environment and Dependency Setup

This phase creates an isolated, stable Python environment with all the correct package versions.

Install System Tools: Install Python's virtual environment tools and the Protocol Buffers compiler.

Generated bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git protobuf-compiler
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Clone the Training Repository:

Generated bash
git clone https://github.com/LeelaChessZero/lczero-training.git
cd lczero-training
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Create a Stable Python Environment with Conda: To resolve TensorFlow version conflicts, we use Conda to create an environment with a compatible Python version.

Generated bash
# (If not installed) Install Miniconda:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate the environment
conda create -n lczero python=3.10
conda activate lczero
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install Core Python Packages: Install TensorFlow and its dependencies. This step addresses the NumPy 2.0 incompatibility by explicitly requiring a version less than 2.0.

Generated bash
pip install --upgrade pip
pip install tensorflow==2.11.* pyyaml "numpy<2"
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install CUDA/cuDNN Libraries: To fix GPU detection issues, install the precise CUDA and cuDNN versions required by TensorFlow 2.11 directly into the Conda environment.

Generated bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Set Environment Variables for GPU Libraries: To ensure TensorFlow can find the newly installed libraries, set the LD_LIBRARY_PATH.

Generated bash
# Set for the current session
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Make the setting permanent for this Conda environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Phase 2: Data Preparation

This phase covers extracting the training data from its archive.

Extract the .tar Archive: The script cannot read the archive directly. Extract the .gz files into a known location.

Generated bash
# Assuming the tar file is in /notebooks/ and we want to extract to /notebooks/
tar -xvf /notebooks/training-run1-test80-20220404-0817.tar -C /notebooks/
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This creates the directory /notebooks/training-run1-test80-20220404-0817/ containing all the training chunk files.

Phase 3: Code Implementation and Troubleshooting

This phase covers the entire iterative process of updating the code, encountering an error, and fixing it.

Problem: Initial attempts to compile the protobuf definition failed due to incorrect file paths.

Solution: All files were consolidated into the tf/ directory. The correct protoc command was identified:

Generated bash
# Run from the lczero-training/tf/ directory
protoc -I=. --python_out=. net.proto
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Note: The final, most robust command was protoc -I=tf --python_out=tf tf/net.proto from the parent directory).

Problem: protoc failed with Enums must contain at least one value and "Net" is already defined.

Diagnosis: The new .proto content was appended to the old content instead of replacing it.

Solution: The tf/net.proto file was completely replaced with a single, correct, merged version containing all necessary definitions and no duplicates.

Problem: Scripts failed with ModuleNotFoundError: No module named 'proto'.

Diagnosis: The protoc command placed net_pb2.py directly in tf/, but the scripts were still using import proto.net_pb2 as pb.

Solution: All import statements in train.py, net.py, and tfprocess.py were changed to import net_pb2 as pb.

Problem: Training failed with FileNotFoundError when trying to load data.

Diagnosis: The fast_chunk_loading mechanism in train.py does not correctly handle path wildcards.

Solution: Added fast_chunk_loading: false to the dataset section of the YAML configuration.

Problem: Training failed with got 0 chunks and IndexError: list index out of range.

Diagnosis: The path in the YAML file ended with /*.gz, which is incorrect for the standard data loader.

Solution: The path in the YAML input was changed to be only the directory path: /notebooks/training-run1-test80-20220404-0817/.

Problem: Training failed with KeyError: 'shuffle_size'.

Diagnosis: The simplified YAML was missing a required parameter for the ChunkParser.

Solution: Added shuffle_size: 800000 and other necessary training parameters back into the YAML file.

Problem: Training failed with AssertionError: 24756 in shufflebuffer.py.

Diagnosis: The ChunkParser was incorrectly calculating the expected size of a data record, mismatching the actual size on disk. The data format was v6, but the parser was assuming a different structure.

Solution: The chunkparser.py file was modified to hard-code the known correct record size (24756) and to use a specific convert_v6_to_tuple function, bypassing the faulty calculation logic.

Problem: Training failed with ValueError: too many values to unpack (expected 31).

Diagnosis: The unpacking logic in convert_v6_to_tuple had the wrong number of variables.

Solution: The tuple unpacking was corrected to expect exactly 32 values, matching the v6_struct definition.

Problem: Training failed with IndexError: list index out of range inside convert_v6_to_tuple.

Diagnosis: The v6 data conversion logic was using data values as list indices, which is unsafe and incorrect for some input formats.

Solution: The convert_v6_to_tuple function was replaced with a robust version that correctly handles all input formats by expanding bytes into bit planes instead of using unsafe indexing.

Problem: Training failed with ResourceExhaustedError (Out of Memory) on the GPU.

Diagnosis: The model and batch size were too large for the GPU's 16 GB of VRAM.

Solution: The batch_size was reduced in the YAML, and eventually, the model size (embedding_size, encoder_layers, etc.) was also reduced to a target of ~9 million parameters to guarantee it would fit.

Problem: The small model still produced Step 0, Loss: nan.

Diagnosis: A fundamental numerical instability in the Mamba block's tf.exp() function, which explodes when using float16 and default random initializers.

Solution: The tfprocess.py was updated to force the Mamba block and all its sub-layers to compute in full float32 precision for stability, and a controlled, non-random initializer was added for the critical A_log weight.

Final State: All software, dependency, path, configuration, and numerical stability issues have been resolved. The final working files are the ones that produced a valid loss at Step 0.

Phase 4: The Final Launch Command

After all prerequisites and fixes are in place, this is the command to start the training.

Activate the environment:

Generated bash
conda activate lczero
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Navigate to the script directory:

Generated bash
cd /notebooks/lczero-training/tf
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Run the training script:

Generated bash
python train.py --cfg configs/mamba2.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
