# TLS certificates

The material referenced by `nginx/sse.conf` lives here and must never be
committed (see `.gitignore` / `.dockerignore`):

- `fullchain.pem` — server certificate plus the intermediate CA chain.
- `privkey.pem` — private key, `chmod 600`.

Self-signed certificate for an internal or staging host:

```sh
openssl req -x509 -newkey rsa:4096 -sha256 -days 825 -nodes \
    -keyout nginx/certs/privkey.pem -out nginx/certs/fullchain.pem \
    -subj "/CN=sse.example.org" \
    -addext "subjectAltName=DNS:sse.example.org"
```

For a publicly trusted certificate issue one with Let's Encrypt (certbot,
cert-manager, ...) and point the two `ssl_*` directives in `nginx/sse.conf` at
the issued files.
