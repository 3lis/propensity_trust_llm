"""
#####################################################################################################################

    trust project - 2025

    list of models in use and their properties

#####################################################################################################################
"""

models                  = (                     # available models (first one is the default)
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "openai/gpt-oss-20b",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M",
        "claude-3-haiku-20240307",              # the less expensive
        "claude-3-5-haiku-20241022",            # cheap
        "claude-3-5-sonnet-20240620",           # medium cost
        "claude-3-7-sonnet-20250219",           # medium cost, the most recent one
        "claude-3-opus-20240229",               # high cost
)
models_interface        = {                     # which interface should be used for a model
        "gpt-3.5-turbo"                         : "openai",
        "gpt-4"                                 : "openai",
        "gpt-4o"                                : "openai",
        "gpt-4o-mini"                           : "openai",
        "gpt-4.1-mini"                          : "openai",
        "openai/gpt-oss-20b"                    : "hf",
        "meta-llama/Llama-2-7b-chat-hf"         : "hf",
        "meta-llama/Llama-2-13b-chat-hf"        : "hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "hf",
        "google/gemma-2-9b-it"                  : "hf",
        "microsoft/Phi-3-mini-4k-instruct"      : "hf",
        "Qwen/Qwen1.5-7B-Chat"                  : "hf",
        "Qwen/Qwen2.5-7B-Instruct"              : "hf",
        "Qwen/Qwen2.5-14B-Instruct-1M"          : "hf",
        "claude-3-haiku-20240307"               : "anthro",
        "claude-3-5-haiku-20241022"             : "anthro",
        "claude-3-5-sonnet-20240620"            : "anthro",
        "claude-3-7-sonnet-20250219"            : "anthro",
        "claude-3-opus-20240229"                : "anthro",
}
models_short_name       = {                     # short name identifying a model, as used in log.txt
        "gpt-3.5-turbo"                         : "gpt35",
        "gpt-4"                                 : "gpt4",
        "gpt-4o"                                : "gpt4o",
        "gpt-4o-mini"                           : "gpt4om",
        "gpt-4.1-mini"                          : "gpt41m",
        "openai/gpt-oss-20b"                    : "gptoss",
        "meta-llama/Llama-2-7b-chat-hf"         : "ll2-7",
        "meta-llama/Llama-2-13b-chat-hf"        : "ll2-13",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "ll3-8",
        "google/gemma-2-9b-it"                  : "gem2-9",
        "microsoft/Phi-3-mini-4k-instruct"      : "ph3m",
        "Qwen/Qwen1.5-7B-Chat"                  : "qw1-7",
        "Qwen/Qwen2.5-7B-Instruct"              : "qw2-7",
        "Qwen/Qwen2.5-14B-Instruct-1M"          : "qw2-14",
        "claude-3-haiku-20240307"               : "cl3h",
        "claude-3-5-haiku-20241022"             : "cl3.5h",
        "claude-3-5-sonnet-20240620"            : "cl3.5s",
        "claude-3-7-sonnet-20250219"            : "cl3.7s",
        "claude-3-opus-20240229"                : "cl3o",
}
models_family           = {                     # which family a model belongs to
        "gpt-3.5-turbo"                         : "openai",
        "gpt-4"                                 : "openai",
        "gpt-4o"                                : "openai",
        "gpt-4o-mini"                           : "openai",
        "gpt-4.1-mini"                          : "openai",
        "openai/gpt-oss-20b"                    : "openai",
        "meta-llama/Llama-2-7b-chat-hf"         : "meta",
        "meta-llama/Llama-2-13b-chat-hf"        : "meta",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "meta",
        "google/gemma-2-9b-it"                  : "google",
        "microsoft/Phi-3-mini-4k-instruct"      : "microsoft",
        "Qwen/Qwen1.5-7B-Chat"                  : "qwen",
        "Qwen/Qwen2.5-7B-Instruct"              : "qwen",
        "Qwen/Qwen2.5-14B-Instruct-1M"          : "qwen",
        "claude-3-haiku-20240307"               : "anthro",
        "claude-3-5-haiku-20241022"             : "anthro",
        "claude-3-5-sonnet-20240620"            : "anthro",
        "claude-3-7-sonnet-20250219"            : "anthro",
        "claude-3-opus-20240229"                : "anthro",
}
