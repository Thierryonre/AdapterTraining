"""SmolVLA LoRA adapter training on MimicGen-style demonstration data."""

import truststore

# Use the OS certificate store for all HTTPS (Hugging Face Hub downloads etc.).
# Needed on machines where antivirus/proxies re-sign TLS traffic.
truststore.inject_into_ssl()
