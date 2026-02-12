document.addEventListener('DOMContentLoaded', () => {
    let radarChartInstance = null;

    // ==========================================
    // 1. PASTEL RED MATRIX RAIN ANIMATION
    // ==========================================
    const canvas = document.getElementById('matrix-bg');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        const str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const matrixChars = str.split("");
        const fontSize = 16;
        let columns = width / fontSize; 
        const drops = [];
        for (let x = 0; x < columns; x++) drops[x] = 1; 

        function drawMatrix() {
            ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = "#ff6961"; 
            ctx.font = fontSize + "px monospace";

            for (let i = 0; i < drops.length; i++) {
                const text = matrixChars[Math.floor(Math.random() * matrixChars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                if (drops[i] * fontSize > height && Math.random() > 0.975) { drops[i] = 0; }
                drops[i]++;
            }
        }
        setInterval(drawMatrix, 45);
    }

    // ==========================================
    // 2. MAIN APP LOGIC
    // ==========================================
    const fileInput = document.getElementById('evidenceInput');
    const imagePreview = document.getElementById('imagePreview');
    const heatmapPreview = document.getElementById('heatmapPreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const dropZone = document.getElementById('dropZone');

    if(dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.backgroundColor = 'rgba(211, 47, 47, 0.2)'; });
        dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.backgroundColor = 'rgba(0,0,0,0.05)'; });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if(e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFileSelect(); }
        });
        fileInput.addEventListener('change', handleFileSelect);
    }

    // --- VISUAL ERROR FEEDBACK ---
    function showDropZoneError(msg) {
        const prompt = document.getElementById('uploadPrompt');
        const originalContent = `
            <i class="bi bi-camera-fill" style="font-size: 3rem;"></i>
            <h4>Drag Evidence Here</h4>
        `;
        
        // Flash Error
        prompt.innerHTML = `
            <i class="bi bi-exclamation-triangle-fill text-danger" style="font-size: 3rem;"></i>
            <h4 class="text-danger fw-bold">${msg}</h4>
        `;
        
        // Reset after 3 seconds
        setTimeout(() => {
            prompt.innerHTML = originalContent;
        }, 3000);
    }

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file) {
            // --- CLIENT-SIDE VALIDATION ---
            if (!file.type.startsWith('image/')) {
                showDropZoneError("INVALID FILE TYPE!");
                alert("⛔ EVIDENCE REJECTED\n\nPlease upload a valid image file (JPG, PNG).");
                fileInput.value = ''; // Reset the input
                imagePreview.style.display = 'none';
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                if(heatmapPreview) heatmapPreview.style.display = 'none';
                document.getElementById('uploadPrompt').style.display = 'none';
                if(analyzeBtn) analyzeBtn.disabled = false;
                document.getElementById('view-toggles').style.display = 'none';
            }
            reader.readAsDataURL(file);
        }
    }

    if(analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const file = fileInput.files[0];
            if (!file) return;

            document.getElementById('action-area').style.display = 'none';
            document.getElementById('loadingArea').style.display = 'block';
            document.getElementById('results-area').style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            fetch('/analyze', { method: 'POST', body: formData })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || "Server Error"); });
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    document.getElementById('verdict-stamp-container').innerText = data.title;
                    document.getElementById('detective-remark').innerText = data.remark;
                    document.getElementById('case-description').innerText = data.description;
                    
                    // --- NEW: UNKNOWN HANDLING ---
                    // Toggle visibility of the CONFIDENCE text based on result
                    const confidenceArea = document.getElementById('confidence-display-area');
                    const verdictConfidence = document.getElementById('verdict-confidence');
                    
                    if (data.title === "UNKNOWN") {
                        // If Unknown: Hide the confidence text entirely
                        if (confidenceArea) confidenceArea.style.display = 'none';
                    } else {
                        // If Valid: Show it and update percentage
                        if (confidenceArea) confidenceArea.style.display = 'block';
                        if (verdictConfidence) verdictConfidence.innerText = data.confidence + "%";
                    }

                    // --- Metrics Updates ---
                    const mAcc = document.getElementById('metric-acc');
                    const mFit = document.getElementById('metric-fit');
                    const mDesc = document.getElementById('metric-fit-desc');

                    if (data.metrics) {
                        if(mAcc) mAcc.innerText = data.metrics.accuracy;
                        if(mFit) {
                            mFit.innerText = "STATUS: " + data.metrics.fit_status;
                            mFit.className = "badge";
                            if(data.metrics.fit_status === "OPTIMAL") mFit.classList.add("bg-success");
                            else if(data.metrics.fit_status === "UNDERFIT") mFit.classList.add("bg-warning", "text-dark");
                            else mFit.classList.add("bg-danger");
                        }
                        if(mDesc) mDesc.innerText = data.metrics.fit_desc;

                        if(data.metrics.matrix_flat) {
                            data.metrics.matrix_flat.forEach((val, index) => {
                                const cell = document.getElementById(`cm-${index}`);
                                if(cell) {
                                    cell.innerText = val.toFixed(2);
                                    const intensity = val * 0.8; 
                                    cell.style.backgroundColor = `rgba(211, 47, 47, ${intensity})`;
                                    cell.style.color = val > 0.5 ? '#fff' : '#000';
                                }
                            });
                        }
                        renderRadarChart(data.metrics.radar_data);
                    }

                    if(heatmapPreview) heatmapPreview.src = "data:image/jpeg;base64," + data.heatmap;
                    document.getElementById('view-toggles').style.display = 'inline-flex';

                    // --- Update Suspect Lineup ---
                    const lineupContainer = document.getElementById('suspect-lineup');
                    lineupContainer.innerHTML = '';
                    
                    data.breakdown.forEach(item => {
                        const opacity = (data.title === "UNKNOWN") ? "0.5" : "1.0"; 
                        const width = item.score + "%";
                        
                        const row = document.createElement('div');
                        row.className = 'lineup-row';
                        row.style.opacity = opacity;
                        row.innerHTML = `
                            <div class="d-flex justify-content-between small fw-bold"><span>${item.label.toUpperCase()}</span><span>${width}</span></div>
                            <div class="lineup-bar-bg"><div class="lineup-bar-fill" style="width: ${width};"></div></div>
                        `;
                        lineupContainer.appendChild(row);
                    });

                    document.getElementById('loadingArea').style.display = 'none';
                    document.getElementById('results-area').style.display = 'block';
                    document.getElementById('download-area').style.display = 'block';
                    setTimeout(() => document.getElementById('verdict-stamp-container').classList.add('stamped'), 100);
                }
            })
            .catch(err => { 
                console.error(err); 
                alert("ACCESS DENIED: " + err.message); 
                location.reload(); 
            });
        });
    }

    function renderRadarChart(radarData) {
        const ctx = document.getElementById('performanceRadar');
        if(!ctx || !radarData) return;
        if (radarChartInstance) radarChartInstance.destroy();

        radarChartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: radarData.labels,
                datasets: radarData.datasets.map((ds, i) => ({
                    label: ds.label,
                    data: ds.data,
                    backgroundColor: 'rgba(211, 47, 47, 0.2)',
                    borderColor: '#d32f2f',
                    pointBackgroundColor: '#fff',
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: '#ccc' },
                        grid: { color: '#ddd' },
                        pointLabels: { font: { family: 'Special Elite', size: 11 } },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // --- FIX: CHECK IF BUTTON EXISTS BEFORE ADDING LISTENER ---
    const dlBtn = document.getElementById('downloadBtn');
    if(dlBtn) {
        dlBtn.addEventListener('click', () => {
            const caseElement = document.getElementById('case-file-panel');
            html2canvas(caseElement, { scale: 2, backgroundColor: null }).then(canvas => {
                const link = document.createElement('a');
                link.download = 'Case_Report.png';
                link.href = canvas.toDataURL();
                link.click();
            });
        });
    }

    // --- ARCHIVES LOGIC ---
    const archiveModal = document.getElementById('archiveModal');
    const archiveTableBody = document.getElementById('archiveTableBody');
    const clearLogsBtn = document.getElementById('clearLogsBtn');

    if (archiveModal && archiveTableBody) {
        archiveModal.addEventListener('show.bs.modal', loadArchives);
    }

    function loadArchives() {
        fetch('/history')
        .then(res => res.json())
        .then(data => {
            archiveTableBody.innerHTML = '';
            if (data.length === 0) {
                archiveTableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">NO RECORDS FOUND</td></tr>';
                return;
            }
            data.forEach(log => {
                const tr = document.createElement('tr');
                let colorClass = "";
                if (log.verdict === "CALCULUS") colorClass = "text-secondary";
                else if (log.verdict === "ULCER") colorClass = "text-danger fw-bold";
                else if (log.verdict === "DISCOLORATION") colorClass = "text-warning fw-bold";
                else colorClass = "text-muted fst-italic";

                tr.innerHTML = `
                    <td>${log.time}</td>
                    <td>${log.id}</td>
                    <td class="text-truncate" style="max-width: 150px;" title="${log.filename || 'N/A'}">${log.filename || '-'}</td>
                    <td class="${colorClass}">${log.verdict}</td>
                    <td>${log.confidence}%</td>
                    <td>${log.match}</td>
                `;
                archiveTableBody.appendChild(tr);
            });
        })
        .catch(err => console.error("Failed to load archives:", err));
    }

    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', () => {
            if(confirm("Are you sure you want to burn all files? This cannot be undone.")) {
                fetch('/clear_history', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    loadArchives();
                });
            }
        });
    }

    // Toggles logic
    const btnShowOriginal = document.getElementById('btn-show-original');
    const btnShowHeatmap = document.getElementById('btn-show-heatmap');
    if(btnShowOriginal) {
        btnShowOriginal.addEventListener('click', (e) => {
            e.stopPropagation();
            imagePreview.style.display = 'block';
            heatmapPreview.style.display = 'none';
            btnShowOriginal.classList.add('active');
            btnShowHeatmap.classList.remove('active');
        });
    }
    if(btnShowHeatmap) {
        btnShowHeatmap.addEventListener('click', (e) => {
            e.stopPropagation();
            imagePreview.style.display = 'none';
            heatmapPreview.style.display = 'block';
            btnShowHeatmap.classList.add('active');
            btnShowOriginal.classList.remove('active');
        });
    }
});

// --- AUTO-UPDATE STATS LOGIC ---
document.addEventListener("DOMContentLoaded", function() {
    const calculusCounter = document.getElementById('count1');
    if (calculusCounter) {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                // Update Total Cases
                if(document.getElementById('total-solved')) {
                    document.getElementById('total-solved').innerText = data.total || 0;
                }
                // Update Breakdown
                if(document.getElementById('count1')) document.getElementById('count1').innerText = data.calculus || 0;
                if(document.getElementById('count2')) document.getElementById('count2').innerText = data.discoloration || 0;
                if(document.getElementById('count3')) document.getElementById('count3').innerText = data.ulcer || 0;
            })
            .catch(err => console.error("Failed to load stats:", err));
    }
});