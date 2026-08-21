#!/usr/bin/env bash
set -euo pipefail

PATCH_NUMPY2=0
if [[ "${1:-}" == "--patch-numpy2" ]]; then
  PATCH_NUMPY2=1
fi

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if [[ "${ID:-}" != "ubuntu" && -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
fi

if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "Ce script cible Ubuntu. Installer MakeHuman manuellement sur cette distribution." >&2
  exit 2
fi

CODENAME="${VERSION_CODENAME:-$(. /etc/os-release && echo "${VERSION_CODENAME}")}"
if [[ "${CODENAME}" != "noble" ]]; then
  echo "Avertissement : le PPA est validé ici pour Ubuntu Noble/24.04, système détecté : ${CODENAME}." >&2
fi

${SUDO} apt-get update
${SUDO} apt-get install -y software-properties-common ca-certificates python3-numpy python3-pip xvfb
${SUDO} add-apt-repository -y ppa:makehuman-official/makehuman-community
${SUDO} apt-get update
${SUDO} apt-get install -y makehuman-community

python3 - <<'PY'
import importlib.util
missing = [m for m in ('numpy', 'trimesh') if importlib.util.find_spec(m) is None]
if missing:
    print('Modules Python manquants:', ', '.join(missing))
    print('Installer avec : python3 -m pip install --user ' + ' '.join(missing))
else:
    print('Modules Python requis présents: numpy, trimesh')
PY

mkdir -p "${HOME}/makehuman/v1py3"
cat > "${HOME}/makehuman/v1py3/socket.cfg" <<'JSON'
{
  "acceptConnections": true,
  "advanced": true,
  "host": "127.0.0.1",
  "port": 12345
}
JSON

echo "MakeHuman installé : $(command -v makehuman-community)"
makehuman-community --help >/tmp/makehuman-community-help.txt 2>&1 || true

echo "Configuration socket : ${HOME}/makehuman/v1py3/socket.cfg"
echo "Lancement test sans écran : xvfb-run -a makehuman-community"

echo "Si NumPy 2 provoque une erreur fromstring/tostring/OpenGL, utiliser --patch-numpy2 après sauvegarde de l’installation ou privilégier le NumPy Ubuntu 1.26."

if [[ "${PATCH_NUMPY2}" -eq 1 ]]; then
  ROOT="/usr/share/makehuman-community"
  if [[ -d "${ROOT}" ]]; then
    ${SUDO} sed -i "s/np\.fromstring(pixels, dtype=np\.uint32)/np.frombuffer(pixels, dtype=np.uint32)/; s/pixels\.tostring()/pixels.tobytes()/g" "${ROOT}/lib/image_qt.py" || true
    ${SUDO} sed -i "s/np\.fromstring(text, dtype='S1')/np.frombuffer(text.encode('utf-8'), dtype='S1')/; s/\.tostring()/\.tobytes()/g" "${ROOT}/core/files3d.py" || true
    echo "Correctifs NumPy 2 appliqués lorsque les fichiers MakeHuman étaient présents."
  fi
fi
