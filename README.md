# submission branch

Apenas os arquivos necessários pra engine da Rinha executar a solução.
O código-fonte fica na branch [`main`](../../tree/main).

## Arquitetura

```
                                  TCP :9999
client ─────────────────────────▶ lb (Rust)
                                    │
                                    │ Unix Domain Socket + SCM_RIGHTS
                                    ▼
                              ┌─────┴─────┐
                              ▼           ▼
                            api1         api2
                         (Rust epoll + IVF AVX2)
```

O LB aceita conexões TCP em 9999 e passa o file descriptor da conexão pra
uma das APIs via `sendmsg(SCM_RIGHTS)`. Round-robin. As APIs recebem o FD
via `recvmsg`, adicionam ao epoll e respondem HTTP/1.1 direto pro cliente.

## Recursos

| Serviço | CPU | RAM |
|---|---|---|
| api1 | 0.4 | 160 MB |
| api2 | 0.4 | 160 MB |
| lb | 0.2 | 30 MB |
| **Total** | **1.0** | **350 MB** |

## Notas

- O `ivf_int16.bin` (87 MB) já vem embutido na imagem `api` em `/data/`.
- O mesmo binário serve como `lb` (via `entrypoint`) ou `api` (default).
- Imagens publicadas em [`smookeydev/rinha-2026-rust-api`](https://hub.docker.com/r/smookeydev/rinha-2026-rust-api).
