"""Execute coding tests only inside a restricted disposable Docker container."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run_tests(response, tests, image, test_imports=(), timeout=15):
    # Caller must supply an existing pinned image. No docker pull/network access.
    if '@sha256:' not in image:
        raise ValueError('Require a digest-pinned, already installed Python image')
    if not shutil.which('docker'):
        return {'scored': False, 'reason': 'docker_unavailable'}
    if subprocess.run(['docker','image','inspect',image], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        return {'scored': False, 'reason': 'pinned_image_not_installed'}
    fenced = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.S)
    code = fenced.group(1) if fenced else response
    with tempfile.TemporaryDirectory(prefix='capability-code-') as directory:
        path = Path(directory)
        # Write the result only after all tests; no host code execution.
        script = '\n'.join(test_imports)+'\n'+code+'\n'+'\n'.join(tests)+'\nprint("CAPABILITY_TESTS_PASSED")\n'
        (path/'case.py').write_text(script)
        command = ['docker','run','--rm','--pull=never','--network=none','--read-only',
                   '--cidfile',str(path/'container.cid'),
                   '--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32',
                   '--memory=256m','--cpus=.5','--user=65534:65534',
                   '--ulimit=cpu=5:5','--ulimit=nofile=64:64',
                   '--mount',f'type=bind,source={path},target=/case,readonly',
                   '--tmpfs=/tmp:rw,noexec,nosuid,size=16m','--entrypoint=python', image,
                   '-I','-B','/case/case.py']
        # Make source readable to the unprivileged container UID, directory stays temporary.
        path.chmod(0o755)
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
            if result.returncode in (125,126,127):
                return {'scored':False,'reason':'container_infrastructure_failure','returncode':result.returncode}
            passed = result.returncode == 0 and result.stdout.rstrip().endswith(b'CAPABILITY_TESTS_PASSED')
            return {'scored': True, 'passed': passed, 'returncode': result.returncode,
                    'scope': 'Provided unit tests only; not proof of arbitrary-code correctness'}
        except subprocess.TimeoutExpired:
            return {'scored': True, 'passed': False, 'reason': 'timeout'}
        finally:
            cid = path/'container.cid'
            if cid.exists():
                identifier = cid.read_text().strip()
                if re.fullmatch(r'[a-f0-9]{64}', identifier):
                    subprocess.run(['docker','rm','-f',identifier], timeout=15,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
