"""Check automatic adapter activation at Python startup and in spawn workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROBE = textwrap.dedent('''\
    import json
    import multiprocessing
    import os
    import sys

    def status():
        return {
            'source': os.environ.get('DSV41_SOURCE'),
            'enabled': any(type(finder).__name__ == 'EngramFinder'
                           for finder in sys.meta_path),
        }

    def worker(connection):
        connection.send(status())
        connection.close()

    if __name__ == '__main__':
        result = {'parent': status()}
        if '--check-worker' in sys.argv:
            context = multiprocessing.get_context('spawn')
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=worker, args=(sender,))
            process.start()
            sender.close()
            assert receiver.poll(15), 'Spawn worker did not report its adapter state'
            result['worker'] = receiver.recv()
            receiver.close()
            process.join(15)
            assert process.exitcode == 0, process.exitcode
        print(json.dumps(result))
''')


class SitecustomizeStartupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / 'launch_probe.py').write_text(PROBE, encoding='utf-8')
        self.environment = dict(os.environ)
        self.environment.pop('DSV41_SOURCE', None)
        adapter = Path(__file__).resolve().parents[1] / 'adapter'
        self.environment['PYTHONPATH'] = os.pathsep.join((str(adapter), str(self.directory)))
        self.environment['PYTHONDONTWRITEBYTECODE'] = '1'

    def launch(self, *arguments):
        result = subprocess.run(
            [sys.executable, '-m', 'launch_probe', *arguments],
            cwd=self.directory, env=self.environment, capture_output=True,
            text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stderr, '')
        return json.loads(result.stdout)

    def test_cli_formats_enable_parent_and_spawn_worker(self):
        model = str(self.directory / 'model with spaces')
        for arguments in (('--model-path', model), ('--model-path=' + model,)):
            with self.subTest(arguments=arguments):
                result = self.launch(*arguments, '--check-worker')
                expected = {'source': model, 'enabled': True}
                self.assertEqual(result, {'parent': expected, 'worker': expected})

    def test_relative_path_is_resolved(self):
        result = self.launch('--model-path', './models/DeepSeek-V4.1-Flash')
        expected = str(self.directory / 'models' / 'DeepSeek-V4.1-Flash')
        self.assertEqual(result['parent'], {'source': expected, 'enabled': True})

    def test_cli_overrides_existing_source(self):
        self.environment['DSV41_SOURCE'] = 'old-model'
        result = self.launch('--model-path=new-model')
        expected = str(self.directory / 'new-model')
        self.assertEqual(result['parent'], {'source': expected, 'enabled': True})

    def test_last_model_path_wins(self):
        result = self.launch('--model-path', 'first', '--model-path=last')
        expected = str(self.directory / 'last')
        self.assertEqual(result['parent'], {'source': expected, 'enabled': True})

    def test_existing_source_enables_parent_and_spawn_worker(self):
        self.environment['DSV41_SOURCE'] = str(self.directory / 'existing-model')
        result = self.launch('--check-worker')
        expected = {'source': self.environment['DSV41_SOURCE'], 'enabled': True}
        self.assertEqual(result, {'parent': expected, 'worker': expected})

    def test_no_model_path_does_not_enable_adapter(self):
        for arguments in ((), ('--', '--model-path', 'ignored'),
                          ('--model-path',), ('--model-path=',),
                          ('--model-path', '--help'), ('--model-path-other=ignored',)):
            with self.subTest(arguments=arguments):
                result = self.launch(*arguments)
                self.assertEqual(result['parent'], {'source': None, 'enabled': False})


if __name__ == '__main__':
    unittest.main()
