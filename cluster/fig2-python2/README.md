# Original Python 2 Figure 2 simulations (`N_e=200`)

This image downloads `delpapa/SORN` at original commit
`cdad55d55f39e04f568ca1bc0c6036bec8db08fb` and runs the author's command:

```text
python test_single.py delpapa.param_Zheng2013
```

The build context deliberately excludes the local checkout, so none of the
Python 3 port is copied into this image. The image always starts from the fresh,
pinned upstream archive.

It generates one five-million-step simulation per container invocation. Run 50
containers with distinct run IDs and seeds to supply panels A-C and the N=200
curve in panels D-E. Other network sizes are still required for complete D-E;
panel F uses hard-coded ranges in the author's plotting script.

## Build and push for NRP

The shared NRP setup used Docker Hub account `seacore219`, namespace
`hengenlab`, and amd64 images. From an Apple Silicon Mac, build and push the
amd64 image directly with Buildx:

```bash
docker login

docker buildx build \
  --platform linux/amd64 \
  --tag seacore219/sorn-fig2-py2:cdad55d \
  --push \
  /path/to/sorn/cluster/fig2-python2
```

The immutable tag is intentional: do not use `latest` for a scientific batch.
If the Docker Hub repository is private, add your existing image pull secret
to `nrp-job.yaml`; a public repository needs no secret.

## Run one simulation

```bash
mkdir -p "$PWD/fig2-python2-results"

docker run --rm \
  --platform linux/amd64 \
  --cpus 1 \
  --memory 3g \
  --env SORN_RUN_ID=n200-run-001 \
  --env SORN_SEED=200001 \
  --volume "$PWD/fig2-python2-results:/opt/sorn/backup" \
  seacore219/sorn-fig2-py2:cdad55d
```

The main result will be written below:

```text
fig2-python2-results/n200-run-001/test_single/<timestamp>/common/result.h5
```

## Run 50 independent jobs on NRP

The supplied manifests follow the previous NRP layout but create a new,
experiment-specific PVC instead of touching either the old sweep storage or
another researcher's storage. The RWX `rook-cephfs` volume permits the indexed
pods to write concurrently.

First inspect the manifests and create the new 20 GiB PVC:

```bash
cd /path/to/sorn/cluster/fig2-python2
kubectl apply -f nrp-pvc.yaml
kubectl get pvc charlesd-sorn-fig2-py2-storage -n hengenlab
```

Wait until its status is `Bound`, then launch the batch:

```bash
kubectl apply -f nrp-job.yaml
kubectl get job charlesd-sorn-fig2-n200 -n hengenlab --watch
```

This is one Kubernetes Indexed Job with 50 completions and at most 10 running
at once. It schedules in the same US West region as `rook-cephfs`. Each pod
automatically maps its zero-based completion index to:

```text
index 0  -> SORN_RUN_ID=n200-run-001, SORN_SEED=200001
index 49 -> SORN_RUN_ID=n200-run-050, SORN_SEED=200050
```

Retries of an index therefore use the same seed and directory. Retry accounting
is per index, so one failing realization does not consume the whole batch's
retry budget. Before applying the Job a second time, either choose a new Job
name and seed base or deliberately reuse the existing outputs; the program does
not silently invent a new seed.

Useful status and diagnostic commands are:

```bash
kubectl get pods -n hengenlab -l job-name=charlesd-sorn-fig2-n200
kubectl logs -n hengenlab -l job-name=charlesd-sorn-fig2-n200 --prefix --tail=30
kubectl describe job charlesd-sorn-fig2-n200 -n hengenlab
```

To verify persistent outputs after completion, create a temporary PVC browser:

```bash
kubectl run fig2-pvc-browser --restart=Never -n hengenlab \
  --image=busybox:1.36 \
  --overrides='{"spec":{"containers":[{"name":"fig2-pvc-browser","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"results","mountPath":"/data"}]}],"volumes":[{"name":"results","persistentVolumeClaim":{"claimName":"charlesd-sorn-fig2-py2-storage"}}]}}'

kubectl exec -n hengenlab fig2-pvc-browser -- \
  find /data -name result.h5
```

Delete only that temporary browser pod when finished; do not delete the PVC,
because it contains the simulations:

```bash
kubectl delete pod fig2-pvc-browser -n hengenlab
```

## Generic array execution

Submit a cluster array with indices 1-50. Derive a stable, unique seed from the
array index; do not use only the launch time because concurrent original-code
processes otherwise receive duplicate seeds.

For a zero-based array index `TASK_INDEX`:

```bash
RUN_NUMBER=$((TASK_INDEX + 1))
RUN_ID=$(printf 'n200-run-%03d' "$RUN_NUMBER")
SEED=$((200000 + RUN_NUMBER))

docker run --rm \
  --cpus 1 \
  --memory 3g \
  --env SORN_RUN_ID="$RUN_ID" \
  --env SORN_SEED="$SEED" \
  --volume /shared/sorn/fig2-python2-results:/opt/sorn/backup \
  seacore219/sorn-fig2-py2:cdad55d
```

The two environment-aware source changes are intentionally limited to output
isolation and random seeding. `param_Zheng2013.py`, `experiment_Zheng2013.py`,
`common/sorn.py`, `common/synapses.py`, and the model equations are unchanged.

Generated `.h5`, pickle, figure, and log files remain on the PVC. The repository
`.gitignore` also excludes those formats if results are copied back locally.
