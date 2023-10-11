# This is repo for volumetric video style transfer

Example cmd:
python --mode style --conf ./confs/hypernerf_nerf_style.conf --case broom --gpu 1 --is_continue --dataset hypernerf --style_num 14

For vscode debug json example， exec on exp_runner_stylize_NGP.py 
{
            "name": "Exp_run_style",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--mode", "style",
                "--conf", "./confs/hypernerf_nerf_style.conf",
                "--case", "toby-sit",
                "--gpu", "1",
                //"--paint",
                "--is_continue",
                "--dataset","hypernerf",
                "--style_num","14",
            ],
            "env": {
                "UNBUFFERED":"1"
            }
        },
