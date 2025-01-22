import subprocess
import os


def discover_declare(input_log_path,output_log_path):
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    subprocess.call(['java', '-version'])
    file_input = input_log_path
    subprocess.call([
        'java', "-Xmx16G",
        '-cp', f'MINERful.jar',
        'minerful.MinerFulMinerStarter',
        "-iLF", file_input,
        "-s", "0.05",
        "-c", "0.98",
        "-g", "0.0",
        "-sT", "0.00",
        "-cT", "0.00",
        "-gT", "0.0",
        '-prune', 'hierarchy',
        '-oJSON', output_log_path
    ], env=env
        , cwd=os.getcwd())


def measurement_extraction(input_log_path,combined_model_path,output_path):
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    subprocess.call(['java', '-version'])
    file_input = input_log_path
    subprocess.call([
        'java', "-Xmx16G",
        '-cp', f'Janus.jar',
        'minerful.JanusMeasurementsStarter',
        "-iLF", file_input,
        "-iME", "json",
        "-iMF", combined_model_path,
        # "-oCSV", output_path,
        "-detailsLevel", "event",
        # "-encodeTasksFlag", "True",
        "-oJSON", output_path
    ], env=env
        , cwd=os.getcwd()+"/Janus-master")