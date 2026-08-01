import sys

from maa.agent.agent_server import AgentServer
from maa.tasker import Tasker

import my_action
import my_reco
import ExpressionRecognition

def main():
    # 设置日志目录（相对 interface.json 所在目录，即 assets/）
    Tasker.set_log_dir("../debug")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)
        
    socket_id = sys.argv[-1]

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
