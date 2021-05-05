import os, tempfile, subprocess, shutil, glob
import distutils.command.clean


def check_for_openmp():
    omp_test = \
        r"""
        #include <omp.h>
        #include <stdio.h>
        int main() {
        #pragma omp parallel
        printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
        }
        """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = 'test.c'
    with open(filename, 'w', buffering=1) as file:
        file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', filename],  stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    return not bool(result)

class clean(distutils.command.clean.clean):
    def run(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(this_dir,'.gitignore'), 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)
        distutils.command.clean.clean.run(self)