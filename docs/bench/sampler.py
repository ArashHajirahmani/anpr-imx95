import json, os, re, socket, subprocess, sys, time, http.client
DURATION = int(sys.argv[1]); TRUTH = "CAKW568"
pid = int(subprocess.check_output(["systemctl", "show", "anpr", "-p", "MainPID", "--value"]).strip())
tck = os.sysconf("SC_CLK_TCK")
class U(http.client.HTTPConnection):
    def __init__(s): super().__init__("localhost"); 
    def connect(s): s.sock = socket.socket(socket.AF_UNIX); s.sock.settimeout(2); s.sock.connect("/run/anpr/api.sock")
def latest():
    c = U(); c.request("GET", "/v1/latest"); r = c.getresponse(); b = r.read(); c.close()
    return json.loads(b) if r.status == 200 else {}
def proc_ticks():
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); return int(f[11]) + int(f[12])
def sys_cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:])); return sum(v), v[3] + v[4]
def temp(z): return int(open(f"/sys/class/thermal/thermal_zone{z}/temp").read()) / 1000
def meminfo():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}; return (m["MemTotal"] - m["MemAvailable"]) / 1024
def rss(): 
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
npu = subprocess.Popen(["timeout", str(DURATION + 1), "neutron_perf.sh"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
start = time.time(); samples = []
pt0, st0 = proc_ticks(), sys_cpu(); t_prev = time.time()
while time.time() - start < DURATION:
    time.sleep(1.0)
    now = time.time(); pt1, st1 = proc_ticks(), sys_cpu()
    d = latest()
    samples.append(dict(t=now - start, proc_cpu=(pt1 - pt0) / tck / (now - t_prev) * 100, sys_cpu=100 * (1 - (st1[1] - st0[1]) / max(1, st1[0] - st0[0])),
        temp_a55=temp(1), temp_ana=temp(0), rss=rss(), mem_used=meminfo(), frame=d.get("frame_id"), fps=d.get("fps"),
        det=d.get("detector_ms"), ocr=d.get("reader_ms"), plates=len(d.get("plates", []))))
    pt0, st0, t_prev = pt1, st1, now
end = time.time()
try: out = npu.communicate(timeout=5)[0]
except Exception: out = ""
npu_load = []; ddr = []
for l in out.splitlines():
    m = re.match(r"\s*(\d+)%\s+(\d+)%\(R\)\s+(\d+)%\(W\)\s+(\d+)%\(T\)", l)
    if m: npu_load.append(int(m.group(1))); ddr.append(int(m.group(4)))
j = subprocess.run(["journalctl", "-u", "anpr", "--no-pager", "--since", f"@{int(start)}", "--until", f"@{int(end)+1}"], capture_output=True, text=True).stdout
cand = re.findall(r"ocr candidate videoCapture text=(\S+) confidence=([\d.]+)", j)
pub = re.findall(r"plate videoCapture text=(\S+) confidence=([\d.]+)", j)
print(json.dumps(dict(duration=end - start, samples=samples, npu_load=npu_load, ddr=ddr, candidates=cand, published=pub)))
