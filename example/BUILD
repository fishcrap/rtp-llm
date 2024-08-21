
py_test(
    name = "test",
    srcs = [
        "test.py",
    ],
    data = [
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
    env = {
        "MODEL_TYPE": "qwen2",
        "CHECKPOINT_PATH": "/home/sxj/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/4d14e384a4b037942bb3f3016665157c8bcb70ea/",
        "TEST_USING_DEVICE": "ARM",
    }
)
