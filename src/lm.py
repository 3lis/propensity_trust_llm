"""
#####################################################################################################################

    HURST project - 2024

    setup of neural language models

#####################################################################################################################
"""

import  os
import  sys
import  datetime
import  platform
import  time
import  string
import  random
import  numpy           as np
from    datasets        import load_dataset, Dataset
from    copy            import deepcopy


LLAMA_VERBOSE           = False

key_file                = "../data/.key.txt"        # file with the current OpenAI API access key
hf_file                 = "../data/.hf.txt"         # file with the current HF API access key
anthro_file             = "../data/.anth.txt"       # file with the current anthropic access key

cnfg                    = None                      # configuration object
client                  = None                      # the language model client object
context_window          = 4096                      # size of the context window in token (default=512)
optimizer               = "paged_adamw_32bit"       # the default optimizer for all training methods

first_ppo_loop          = True
first_ppo_lora          = None

# ===================================================================================================================
#
#   Basic utilities
#
#       set_hf
#       set_openai
#       set_anthro
#       ls_models
#       ft_status
#       save_backup
#       load_backup
#
# ===================================================================================================================


def set_hf():
    """
    Get the hugginface client

    NOTE: this is the first function to call before all others using hugginface models
    """
    from    transformers    import pipeline, AutoTokenizer, LlamaTokenizerFast, AutoModelForCausalLM
    from    huggingface_hub import login
    import  torch

    client          = None
    key             = open( hf_file, 'r' ).read().rstrip()
    login( token=key )


    match cnfg.model:
        # there are models that needs to extract their tokenizer
        case "microsoft/Phi-3-mini-4k-instruct"     |\
             "meta-llama/Llama-2-7b-chat-hf"        |\
             "meta-llama/Llama-2-13b-chat-hf"       |\
             "Qwen/Qwen1.5-7B-Chat"                 |\
             "Qwen/Qwen2.5-7B-Instruct"             |\
             "Qwen/Qwen2.5-14B-Instruct-1M":
            tokenizer       = AutoTokenizer.from_pretrained( cnfg.model )
            client          = pipeline(
                task        = "text-generation",
                model       = cnfg.model,
                tokenizer   = tokenizer,
                device_map  = "auto",
                torch_dtype = torch.bfloat16,
                model_kwargs= { "low_cpu_mem_usage": True }
            )
            return client

        # others don't
        case "meta-llama/Meta-Llama-3.1-8B-Instruct" |\
             "openai/gpt-oss-20b":
            client          = pipeline(
                task        = "text-generation",
                model       = cnfg.model,
                device_map  = "auto",
                torch_dtype = torch.bfloat16,
                model_kwargs= { "low_cpu_mem_usage": True }
            )
            return client

        # take care of TorchDynamo that crashes easily...
        case "google/gemma-2-9b-it":
            torch._dynamo.config.disable = True
            torch._dynamo.reset()
            tokenizer       = AutoTokenizer.from_pretrained( cnfg.model )
            client          = pipeline(
                task        = "text-generation",
                model       = cnfg.model,
                tokenizer   = tokenizer,
                device_map  = "auto",
                torch_dtype = torch.bfloat16,
                model_kwargs= { "low_cpu_mem_usage": True }
            )
            return client


    if cnfg.VERBOSE:
        print( f"\n\nerror in set_hf(): {cnfg.model} not supported\n\n" )
    sys.exit()


def set_openai():
    """
    Parse the key to OpenAI

    NOTE: this is the first function to call before all others using OpenAI
    """
    from    openai          import OpenAI
    key             = open( key_file, 'r' ).read().rstrip()
    client          = OpenAI( api_key=key )
    return client


def set_anthro():
    """
    Parse the anthropic key and return the client
        NOTE: should be the first function to call before all others that use anthropic
    """
    import anthropic
    key             = open( anthro_file, 'r' ).read().rstrip()
    client          = anthropic.Anthropic( api_key=key )
    return client


def set_multiple_lm():
    """
    set clients for multiple language models, associated with agents

    NOTE: this function requires a valid cnfg.agent_models dictionary
    """
    assert isinstance( cnfg.agent_models, dict ), "cannot set multiple models without cnfg.agent_models"

    clients = dict()                # keep trace of instantiated clients, to avoid duplicate

    for a in cnfg.agent_models.keys():
        model_id    = cnfg.agent_models[ a ][ "model_id" ]
        # if there is already a client for this model_id, use it
        if model_id in clients.keys():         
            cnfg.agent_models[ a ][ "client" ]  = clients[ model_id ]
            continue
        # otherwise instantiate a fresh new one
        cnfg.model_id   = model_id
        cnfg.model      = cnfg.agent_models[ a ][ "model" ]
        cnfg.mode       = cnfg.agent_models[ a ][ "mode" ]
        cnfg.interface  = cnfg.agent_models[ a ][ "interface" ]
        if cnfg.interface == "openai":
            client      = set_openai()
        if cnfg.interface == "hf":
            client      = set_hf()
        clients[ model_id ] = client
        cnfg.agent_models[ a ][ "client" ]  = client


def ls_models():
    """
    List available models and uploaded files (OpenAI)
    """
    global client

    if client is None:
        # check if openai has already a client, otherwise set it
        client  = set_openai()

    res         = client.files.list()
    print( "\n\nlist of all uploaded files:" )
    res         = res.to_dict()[ "data" ]
    for f in res:
        t   = datetime.datetime.fromtimestamp( f[ "created_at" ], datetime.UTC )
        print( f[ "id" ], '\t', t, '\t', f[ "filename" ] )

    res         = client.models.list().data
    print( "\n\nlist of all available models:" )
    for m in res:
        t   = datetime.datetime.fromtimestamp( m.created, datetime.UTC )
        print( m.id, '\t', t )
    return res
