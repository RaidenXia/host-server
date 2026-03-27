import dashscope

class Settings:
    protocol_version = 2
    # LLM API 密钥
    dashscope.api_key = "sk-0b72ca76ba5e486d984c316fc097e67c"

    # 模型选择
    INTENT_MODEL = "qwen-turbo"       # 专门用于意图识别
    CHAT_MODEL = "qwen-turbo"         # 用于常规对话

    # device
    ASR_DEVICE = "cuda"            # ASR 模型使用的设备
    # ASR_DEVICE = "cuda"            # ASR 模型使用的设备
    VAD_DEVICE = "cuda"            # VAD 模型使用的设备

    # 超时设置
    API_TIMEOUT = 10  # 秒

    def Set_API_Key(self, aliyun_api_key):
        dashscope.api_key = aliyun_api_key

global_settings = Settings()
