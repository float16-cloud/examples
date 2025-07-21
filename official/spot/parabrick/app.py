import subprocess

ROOT_DIR = "/apps/workspace"

command = [
    "pbrun", "fq2bam",
    "--ref", f"{ROOT_DIR}/parabricks_sample/Ref/Homo_sapiens_assembly38.fasta",
    "--in-fq", f"{ROOT_DIR}/parabricks_sample/Data/sample_1.fq.gz", f"{ROOT_DIR}/parabricks_sample/Data/sample_2.fq.gz",
    "--out-bam", f"{ROOT_DIR}/fq2bam_output.bam",
    "--num-gpus", "1"
]

try:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Command executed successfully")
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred while executing the command")
    print("Error output:", e.stderr)