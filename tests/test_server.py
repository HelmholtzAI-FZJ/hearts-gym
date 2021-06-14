import subprocess
import time
import unittest

from hearts_gym.envs import hearts_server
from hearts_gym.envs import server_utils


class TestCommon(unittest.TestCase):
    def test_encode_decode(self):
        original = {
            'test': 0,
            'test2': [1, 2, 3],
            'test3': {},
        }
        encoded = server_utils.encode_data(original)
        decoded = server_utils.decode_data(encoded)
        self.assertEqual(original, decoded)

    def test_runs_and_quits(self):
        port = hearts_server.PORT + 0

        server_proc = subprocess.Popen(
            [
                'python',
                'start_server.py',
                '--num_procs',
                '1',
                '--num_parallel_games',
                '16',
                '--max_num_games',
                '256',
                '--mask_actions',
                'True',
                '--seed',
                '0',
                '--port',
                str(port),
            ],
        )
        time.sleep(2)
        self.assertIsNone(server_proc.poll())

        client_procs = []
        for i in range(4):
            client_proc = subprocess.Popen(
                [
                    'python',
                    'eval_agent.py',
                    (
                        'results/PPO/PPO_Hearts-v0_ff23b_00000_0_'
                        '2021-06-02_18-45-54/checkpoint_000120/checkpoint-120/'
                    ),
                    '--name',
                    b'\xf0\x9f\x92\xa9'.decode(),
                    '--algorithm',
                    'PPO',
                    '--framework',
                    'tf',
                    '--port',
                    str(port),
                ],
            )
            client_procs.append(client_proc)

        try:
            client_procs[0].wait(60)
        except subprocess.TimeoutExpired:
            pass
        for client_proc in client_procs[1:]:
            try:
                client_proc.wait(5)
            except subprocess.TimeoutExpired:
                pass
            client_proc.terminate()

        server_proc.terminate()
        try:
            server_proc.wait(5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            try:
                server_proc.wait(5)
            except subprocess.TimeoutExpired:
                print('server process still running...')

        for client_proc in client_procs:
            client_proc.kill()
            try:
                client_proc.wait(5)
            except subprocess.TimeoutExpired:
                print('client process still running...')


if __name__ == '__main__':
    unittest.main()
