<p align="center">
  <h1 align="center">rinha-2026-rust</h1>
  <p align="center">Backend Rust para a Rinha de Backend 2026. k-NN k=5 sobre 3M vetores em 14 dimensões para detecção de fraude.</p>
  <p align="center">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/rust-1.92-red.svg" alt="Rust">
    <img src="https://img.shields.io/badge/alpine-3.20-blue.svg" alt="Alpine">
    <img src="https://img.shields.io/badge/score-6000%2F6000-brightgreen.svg" alt="Score">
    <img src="https://img.shields.io/badge/detection-100%25-brightgreen.svg" alt="Detection">
    <img src="https://img.shields.io/badge/platform-linux--amd64-lightgrey.svg" alt="Platform">
  </p>
</p>

---

## Sumário

- [Sobre](#sobre)
- [Resultado](#resultado)
- [Como funciona](#como-funciona)
- [Otimizações](#otimizações)
- [Requisitos](#requisitos)
- [Como rodar](#como-rodar)
- [Verificação offline](#verificação-offline)
- [Detalhes técnicos](#detalhes-técnicos)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Inspirações](#inspirações)
- [Licença](#licença)

## Sobre

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), cujo desafio é construir uma API de detecção de fraude em transações de cartão usando busca vetorial sobre 3 milhões de referências. Cada requisição precisa ser normalizada em 14 dimensões, classificada via k-NN k=5 e respondida em tempo hábil.

A solução roda inteira em Rust, com o load balancer fazendo FD-passing via `sendmsg(SCM_RIGHTS)` para as APIs e uma busca IVF (Inverted File Index) com 4.096 clusters acelerada por AVX2.

## Resultado

Rodando o `test/test.js` oficial do upstream (`k6 run`, 900 req/s por 2 min) num Ryzen 5700X com carga de fundo:

| Métrica | Valor |
|---|---|
| `p99` | 0.87 a 1.10 ms |
| `final_score` | 5959 a 6000 (4 de 5 runs cravam 6000) |
| `detection_score` | 3000 (perfeito) |
| `failure_rate` | 0.00% |
| `FP / FN / Err` | 0 / 0 / 0 |

A pontuação de detecção é determinística: todas as execuções dão 3000 cravado. A variância vem só da latência de cauda, que num host limpo (o Mac Mini do teste oficial) deve ficar estável abaixo de 1 ms.

## Como funciona

```
TCP :9999                                 SCM_RIGHTS sobre UDS
client ─────▶ lb (Rust, ~250 linhas) ─────────────────────────┐
                                                              ▼
                                                       ┌────────────┐
                                                       │ api1, api2 │
                                                       │ epoll +    │
                                                       │ IVF AVX2   │
                                                       └────────────┘
```

O LB aceita TCP na porta 9999 e passa o file descriptor do socket para uma das APIs via `sendmsg(SCM_RIGHTS)` numa Unix Stream Socket persistente (round-robin). A API recebe o FD via `recvmsg`, joga no epoll e responde HTTP/1.1 direto pro cliente. Nenhum byte do payload passa pelo LB.

### Pipeline da requisição

1. **LB**: `accept4()` no socket TCP, escolhe upstream via `AtomicUsize::fetch_add` round-robin.
2. **LB**: `sendmsg(SCM_RIGHTS, [client_fd])` sobre a conexão UDS persistente com aquele upstream.
3. **API thread `fd-recv`**: `recvmsg` extrai o FD do `cmsg`, manda para `mpsc::channel`, escreve `1` no `eventfd`.
4. **API thread `epoll`**: acorda no eventfd, drena a channel, adiciona o FD ao `epoll` com `EPOLLIN`.
5. **API thread `epoll`**: `EPOLLIN` dispara, lê HTTP, parseia JSON, normaliza nas 14 dims, quantiza para `i16`.
6. **API thread `epoll`**: roda `fraud_count()` (centroid scan AVX2 + cluster scan AVX2), pega resposta pré-formatada pelo `fraud_count`, escreve com `send(MSG_NOSIGNAL)`.

## Otimizações

| Camada | O quê |
|---|---|
| Storage | 87 MB Int16 SoA (vs 168 MB Float32) com align 2 MB + `MADV_HUGEPAGE` |
| Centroids | Scan SoA com AVX2 + FMA, 8 clusters por iteração (4.096 × 14, 512 panels) |
| Cluster scan | `_mm256_cvtepi16_epi32` + `_mm256_cvtepi32_ps` na hora, 8 vetores por panel |
| Top-K | Insertion sort com early skip via `_mm_min_ss` sobre o panel atual |
| Servidor | epoll level-triggered, thread principal pinada via `sched_setaffinity` |
| Sockets | `SO_BUSY_POLL=50us` + `TCP_NODELAY` por client conn |
| LB → API | `SCM_RIGHTS` (zero proxy) com 2 workers `SO_REUSEPORT` |
| HTTP/JSON | Parsers manuais zero-alloc, respostas HTTP pré-formatadas |

## Requisitos

- **OS**: Linux amd64 (alvo: Mac Mini Late 2014 com Ubuntu 24.04)
- **CPU**: Haswell ou superior (AVX2 + FMA + F16C obrigatórios)
- **Build**: Rust 1.92, Docker 24+ com Compose v2
- **Teste de carga**: [k6](https://k6.io/) (opcional, para rodar o cenário oficial)

## Como rodar

### Subir a stack local

```bash
docker compose up --build -d
curl http://localhost:9999/ready
```

A imagem é multi-stage Alpine e embute o `ivf_int16.bin` (87 MB) em `/data/`, então não há dependência externa.

### Rodar o teste oficial

Clonar o upstream e rodar o `k6`:

```bash
git clone https://github.com/zanfranceschi/rinha-de-backend-2026
cd rinha-de-backend-2026
k6 run test/test.js
jq . test/results.json
```

### Recursos do `docker-compose`

| Serviço | CPU | RAM |
|---|---|---|
| `lb` | 0.2 | 30 MB |
| `api1` | 0.4 | 160 MB |
| `api2` | 0.4 | 160 MB |
| **Total** | **1.0** | **350 MB** |

Encosta no limite oficial de 1 CPU / 350 MB.

## Verificação offline

Para checar a accuracy sem subir a stack inteira:

```bash
cargo build --release --bin verify
./target/release/verify ivf_int16.bin /caminho/pra/test-data.json 192
```

Esperado:

```
correct: 54100/54100 (100.0000%)
errors:  0 (FP=0 FN=0, E=0)
score_det estimate: 3000.00
```

## Detalhes técnicos

### Quantização Int16

Os vetores das 3M referências são quantizados de Float32 para Int16 com `scale=10000`. Como o `references.json` oficial já vem com `round4` aplicado, essa quantização é **lossless**: cada valor float `v` vira o inteiro `round(v * 10000)` sem perda.

Memória: `3.000.000 × 14 × 2 bytes = 84 MB` (vs 168 MB em Float32). Cabe folgado no limite de 160 MB por API.

### Layout SoA + AVX2

Dentro de cada cluster, os vetores são empacotados em panels SoA de 8:

```
panel = [v0d0, v1d0, ..., v7d0, v0d1, v1d1, ..., v7d1, ..., v0d13, ..., v7d13]
```

O scan AVX2 processa 8 vetores por iteração: para cada dimensão `d`, carrega 16 bytes (8 × `i16`), expande para 8 × `f32` via `_mm256_cvtepi16_epi32` + `_mm256_cvtepi32_ps`, subtrai a query broadcast e acumula com FMA. Após 14 dimensões, tem 8 distâncias L² simultâneas no acumulador.

### IVF e nprobe

Índice IVF treinado com FAISS:

- `nlist = 4096` clusters
- `nprobe = 192` clusters varridos por query

Com `nprobe = 192`, o IVF reproduz com fidelidade de 100% o resultado do brute-force k-NN k=5 sobre todas as 54.100 entries do `test-data.json` oficial.

### FD-passing

Padrão herdado do `gabrielrauch/lb` e do `silent-index/fd_lb.c`. O LB nunca lê os bytes do payload, só o `SOCK_STREAM` Unix Domain Socket persistente serve para transportar o FD do cliente via `cmsg`:

```c
sendmsg(control_fd, &msg, MSG_NOSIGNAL);
// msg.msg_control aponta para um cmsghdr com SCM_RIGHTS + client_fd
```

A API recebe via `recvmsg` e o kernel duplica o FD na tabela de descriptors do processo destino. Como não há proxy de bytes, evitamos uma cópia kernel-to-userspace-to-kernel e um overhead consistente de 300 a 500 μs (vs HAProxy).

### Threading

Cada container API roda 3 threads:

- `fd-accept`: blocking `accept4` no UDS listener; cria 1 thread `fd-recv` por LB control conn.
- `fd-recv`: blocking `recvmsg` loop, empurra FDs para `mpsc::channel`, sinaliza via `eventfd`.
- `epoll` (thread principal): atende as conexões HTTP, é pinada via `sched_setaffinity` na primeira CPU permitida pelo cgroup.

### Compilação

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

E `rustflags = ["-C", "target-cpu=haswell"]` no `.cargo/config.toml`.

## Estrutura do projeto

```
.
├── Dockerfile              # Multi-stage Alpine, embute o ivf_int16.bin
├── docker-compose.yml      # 3 serviços: api1, api2, lb
├── ivf_int16.bin           # 87 MB, índice IVF pré-construído
├── info.json               # Metadados da submissão
├── src/
│   ├── main.rs             # Entrypoint da API
│   ├── server.rs           # epoll + recvmsg(SCM_RIGHTS)
│   ├── ivf.rs              # Carga + inferência IVF AVX2
│   ├── normalize.rs        # 14 dims (espelha o gerador C)
│   ├── http.rs             # Parser HTTP/1.1 zero-alloc
│   ├── json.rs             # Parser JSON zero-alloc
│   ├── response.rs         # 6 respostas pré-formatadas
│   └── bin/
│       ├── lb.rs           # Load balancer FD-passing
│       └── verify.rs       # Validação offline (54100 entries)
└── tools/                  # Scripts Python para regerar o ivf_int16.bin
```

A branch [`submission`](../../tree/submission) contém apenas o `docker-compose.yml` e o `info.json`, referenciando as imagens publicadas no Docker Hub. É a branch que a engine da Rinha executa.

## Inspirações

A arquitetura de FD-passing veio de olhar dois dos top 3 da Rinha:

- [@gabrielrauch/rinha-2026](https://github.com/gabrielrauch/rinha-2026): LB Rust com `SO_REUSEPORT`, threads de recv por control conn, mpsc + monoio.
- [@crepao-da-massa/silent-index](https://github.com/crepao-da-massa/silent-index): LB em C puro com `cmsg` pré-inicializado e `MSG_NOSIGNAL`.

Os dois usam o mesmo padrão (LB próprio + `sendmsg(SCM_RIGHTS)`) e foi o caminho que tirou ~300 μs do overhead vs HAProxy no meu setup.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
