def print_model_memory_footprint(model):
    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"The model has {num_parameters:,} parameters.")

    num_tensors = sum(t.numel() for t in model.buffers())
    print(f"The model has {num_tensors:,} tensors.")

    memory_bytes = (num_parameters + num_tensors) * 4  # for float32
    memory_kb = memory_bytes / 1024  # KB
    memory_mb = memory_kb / 1024  # MB

    print(f"The model's memory footprint is approximately {memory_mb:.2f} MB.")
