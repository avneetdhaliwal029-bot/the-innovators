document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropPrompt = document.getElementById('drop-zone-prompt');
  const previewWrapper = document.getElementById('preview-wrapper');
  const imagePreview = document.getElementById('image-preview');
  const removeBtn = document.getElementById('remove-btn');
  const analyzeBtn = document.getElementById('analyze-btn');
  const btnText = analyzeBtn.querySelector('.btn-text');
  const btnSpinner = analyzeBtn.querySelector('.btn-spinner');
  const errorBox = document.getElementById('error-box');
  const errorMessage = document.getElementById('error-message');
  const resultPanel = document.getElementById('result-panel');
  const resultConfident = document.getElementById('result-confident');
  const resultUncertain = document.getElementById('result-uncertain');
  const binBanner = document.getElementById('bin-banner');
  const binName = document.getElementById('bin-name');
  const disposalTip = document.getElementById('disposal-tip');
  const tipCard = document.getElementById('tip-card');
  const categoryBadge = document.getElementById('category-badge');
  const confidenceTag = document.getElementById('confidence-tag');
  const probabilityList = document.getElementById('probability-list');
  const uploadSection = document.querySelector('.upload-section');
  const actionBar = document.querySelector('.action-bar');
  const singleModeBtn = document.getElementById('single-mode-btn');
  const bulkModeBtn = document.getElementById('bulk-mode-btn');

  // Create Bulk UI elements
  const bulkPreviewGrid = document.createElement('div');
  bulkPreviewGrid.id = 'bulk-preview-grid';
  bulkPreviewGrid.className = 'bulk-preview-grid';
  bulkPreviewGrid.hidden = true;
  uploadSection.insertBefore(bulkPreviewGrid, actionBar);

  const analyzeAllBtn = document.createElement('button');
  analyzeAllBtn.type = 'button';
  analyzeAllBtn.id = 'analyze-all-btn';
  analyzeAllBtn.className = 'btn btn-primary';
  analyzeAllBtn.disabled = true;
  analyzeAllBtn.textContent = 'Analyze All Images';
  analyzeAllBtn.hidden = true;
  actionBar.appendChild(analyzeAllBtn);

  const bulkResultContainer = document.createElement('div');
  bulkResultContainer.className = 'bulk-result-container';
  bulkResultContainer.style.display = 'flex';
  bulkResultContainer.style.flexDirection = 'column';
  bulkResultContainer.style.gap = '1.25rem';
  bulkResultContainer.style.marginTop = '0.5rem';
  bulkResultContainer.hidden = true;
  resultPanel.parentNode.insertBefore(bulkResultContainer, resultPanel.nextSibling);

  // State
  let currentFile = null;
  let isAnalyzing = false;
  let isBulkMode = false;
  let bulkFiles = [];

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.webp'];
  const MAX_FILE_SIZE = 8 * 1024 * 1024; // 8 MB

  // --- Modes ---
  function switchToSingleMode() {
    isBulkMode = false;
    bulkModeBtn.classList.remove('active');
    singleModeBtn.classList.add('active');
    fileInput.multiple = false;
    bulkPreviewGrid.hidden = true;
    previewWrapper.hidden = true;
    analyzeAllBtn.hidden = true;
    analyzeBtn.hidden = false;
    analyzeAllBtn.disabled = true;
    bulkFiles = [];
    resultPanel.hidden = true;
    bulkResultContainer.hidden = true;
    dropPrompt.hidden = false;
    clearError();
  }

  function switchToBulkMode() {
    isBulkMode = true;
    singleModeBtn.classList.remove('active');
    bulkModeBtn.classList.add('active');
    fileInput.multiple = true;
    bulkPreviewGrid.hidden = false;
    previewWrapper.hidden = true;
    analyzeAllBtn.hidden = false;
    analyzeBtn.hidden = true;
    analyzeAllBtn.disabled = true;
    resultPanel.hidden = true;
    bulkResultContainer.hidden = true;
    dropPrompt.hidden = false;
    clearError();
  }

  singleModeBtn.addEventListener('click', switchToSingleMode);
  bulkModeBtn.addEventListener('click', switchToBulkMode);

  // --- Helpers ---
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBox.hidden = false;
    errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    errorBox.hidden = true;
    errorMessage.textContent = '';
  }

  function validateFile(file) {
    if (!file) {
      showError('Please select an image file to proceed.');
      return false;
    }
    const name = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTS.some(ext => name.endsWith(ext));
    const hasValidMime = ALLOWED_TYPES.includes(file.type);

    if (!hasValidExt && !hasValidMime) {
      showError('Unsupported image format. Please upload a JPG, JPEG, PNG, or WebP image.');
      return false;
    }
    if (file.size > MAX_FILE_SIZE) {
      showError('Image size must be less than 8 MB.');
      return false;
    }
    return true;
  }

  function getColorForClass(cls) {
    const classColors = {
      cardboard: '#7C4A21',
      glass: '#2F855A',
      metal: '#4A6572',
      paper: '#2B6CB0',
      plastic: '#A16207',
      trash: '#2D3748'
    };
    return classColors[cls] || '#718096';
  }

  // --- Single File Handling ---
  function handleFileSelection(file) {
    clearError();
    resultPanel.hidden = true;
    bulkResultContainer.hidden = true;
    if (!validateFile(file)) {
      resetSelection();
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.hidden = true;
      previewWrapper.hidden = false;
      analyzeBtn.disabled = false;
    };
    reader.onerror = () => {
      showError('Failed to read image file. Please try again with a valid photo.');
      resetSelection();
    };
    reader.readAsDataURL(file);
  }

  function resetSelection() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropPrompt.hidden = false;
    previewWrapper.hidden = true;
    analyzeBtn.disabled = true;
    resultPanel.hidden = true;
  }

  // --- Bulk File Handling ---
  function handleBulkFiles(files) {
    clearError();
    resultPanel.hidden = true;
    bulkResultContainer.hidden = true;
    
    // Add new files to the array
    for (const file of files) {
      if (validateFile(file)) {
        bulkFiles.push(file);
      }
    }
    renderBulkPreview();
    
    // Reset fileInput so the same file can be selected again
    fileInput.value = '';
  }

  function renderBulkPreview() {
    bulkPreviewGrid.innerHTML = '';
    
    if (bulkFiles.length > 0) {
      dropPrompt.hidden = true;
      analyzeAllBtn.disabled = false;
    } else {
      dropPrompt.hidden = false;
      analyzeAllBtn.disabled = true;
    }

    bulkFiles.forEach((file, index) => {
      const card = document.createElement('div');
      card.className = 'bulk-card';
      
      const img = document.createElement('img');
      img.className = 'bulk-image';
      img.alt = file.name;
      img.src = URL.createObjectURL(file);
      
      const nameSpan = document.createElement('span');
      nameSpan.className = 'bulk-filename';
      nameSpan.textContent = file.name;
      
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'bulk-remove-btn';
      removeBtn.innerHTML = '&times;';
      removeBtn.setAttribute('aria-label', `Remove ${file.name}`);
      
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        bulkFiles.splice(index, 1);
        renderBulkPreview();
      });
      
      card.appendChild(img);
      card.appendChild(nameSpan);
      card.appendChild(removeBtn);
      bulkPreviewGrid.appendChild(card);
    });
  }

  // --- Drop zone and Input Events ---
  dropZone.addEventListener('click', (e) => {
    if (e.target === removeBtn || removeBtn.contains(e.target) || e.target.classList.contains('bulk-remove-btn')) {
      return;
    }
    fileInput.click();
  });

  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (isBulkMode) {
      if (e.target.files && e.target.files.length > 0) {
        handleBulkFiles(Array.from(e.target.files));
      }
    } else {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    }
  });

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetSelection();
    clearError();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      if (isBulkMode) {
        handleBulkFiles(Array.from(dt.files));
      } else {
        handleFileSelection(dt.files[0]);
      }
    }
  });

  // --- API calls ---
  analyzeBtn.addEventListener('click', async () => {
    if (isAnalyzing || !currentFile) return;

    clearError();
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnSpinner.hidden = false;
    resultPanel.hidden = true;
    bulkResultContainer.hidden = true;

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || 'Upload failed. Please try again.');
        return;
      }

      displayResult(data);
    } catch (err) {
      showError('Upload failed. Please try again.');
    } finally {
      isAnalyzing = false;
      analyzeBtn.disabled = !currentFile;
      btnText.textContent = 'Analyze Waste Item';
      btnSpinner.hidden = true;
    }
  });

  analyzeAllBtn.addEventListener('click', async () => {
    if (isAnalyzing || bulkFiles.length === 0) return;
    
    clearError();
    isAnalyzing = true;
    analyzeAllBtn.disabled = true;
    analyzeAllBtn.textContent = 'Analyzing...';
    resultPanel.hidden = true;
    bulkResultContainer.hidden = false;
    bulkResultContainer.innerHTML = ''; // Clear old results

    for (const file of bulkFiles) {
      const formData = new FormData();
      formData.append('image', file);
      try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) {
          showError(data.error || 'Upload failed for a file. Please try again.');
          continue;
        }
        displayBulkResult(file, data);
      } catch (err) {
        showError('Upload failed. Please try again.');
      }
    }
    
    isAnalyzing = false;
    analyzeAllBtn.textContent = 'Analyze All Images';
    analyzeAllBtn.disabled = false;
  });

  function displayResult(data) {
    resultPanel.hidden = false;
    const { label, confidence, uncertain, bin, color, tip, scores } = data;

    if (uncertain) {
      resultConfident.hidden = true;
      resultUncertain.hidden = false;
    } else {
      resultConfident.hidden = false;
      resultUncertain.hidden = true;

      binBanner.style.backgroundColor = color || '#2b6cb0';
      binName.textContent = bin || 'Recycling';
      categoryBadge.textContent = label;
      confidenceTag.textContent = `${Math.round(confidence * 100)}% match`;
      disposalTip.textContent = tip || 'Dispose responsibly according to local guidelines.';
      tipCard.style.borderLeftColor = color || '#2b6cb0';
    }

    // Populate probability breakdown
    probabilityList.innerHTML = '';
    if (Array.isArray(scores)) {
      scores.forEach(item => {
        const pct = Math.round(item.score * 100);
        const itemRow = document.createElement('div');
        itemRow.className = 'prob-item';
        itemRow.innerHTML = `
          <div class="prob-label-row">
            <span class="prob-name">${item.label}</span>
            <span class="prob-pct">${pct}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${pct}%; background-color: ${getColorForClass(item.label)};"></div>
          </div>
        `;
        probabilityList.appendChild(itemRow);
      });
    }

    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function displayBulkResult(file, data) {
    const card = document.createElement('div');
    card.className = 'result-card';
    const { label, confidence, uncertain, bin, color, tip, scores } = data;
    
    if (uncertain) {
      card.innerHTML = `
        <div class="result-header">
          <span class="badge" id="category-badge">Uncertain</span>
        </div>
        <img src="${URL.createObjectURL(file)}" class="result-image" alt="Uncertain" style="max-width: 100%; border-radius: 8px; margin-bottom: 1rem;"/>
        <div class="uncertain-card" style="border:none; padding:0; box-shadow:none;">
          <h2 class="uncertain-title">Not Sure About This Item</h2>
          <p class="uncertain-text">
            Our classifier confidence was below 50%. We cannot reliably recommend a bin for this photo.
          </p>
        </div>
      `;
    } else {
      card.innerHTML = `
        <div class="result-header">
          <span class="badge" id="category-badge">${label}</span>
          <span class="confidence-tag" id="confidence-tag">${Math.round(confidence * 100)}% match</span>
        </div>
        <img src="${URL.createObjectURL(file)}" class="result-image" alt="${label}" style="max-width: 100%; border-radius: 8px; margin-bottom: 1rem;"/>
        <div class="bin-banner" style="background-color:${color};">
          <div class="bin-details">
            <span class="bin-label">Target Bin</span>
            <h2 class="bin-name">${bin}</h2>
          </div>
        </div>
        <div class="tip-card">
          <h3 class="tip-title">Disposal Tip</h3>
          <p class="tip-body">${tip}</p>
        </div>
        <div class="probability-list" style="margin-top: 1rem;">
          <h3 style="font-size: 0.85rem; margin-bottom: 0.5rem;">Class Probabilities</h3>
          ${scores.map(item => `<div class="prob-item"><div class="prob-label-row"><span class="prob-name">${item.label}</span><span class="prob-pct">${Math.round(item.score*100)}%</span></div><div class="prob-bar-track"><div class="prob-bar-fill" style="width:${Math.round(item.score*100)}%; background-color:${getColorForClass(item.label)};"></div></div></div>`).join('')}
        </div>
      `;
    }
    
    bulkResultContainer.appendChild(card);
  }

  // --- PWA handling ---
  let deferredPrompt;
  const installBtn = document.getElementById('install-btn');
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (installBtn) installBtn.hidden = false;
  });
  
  if (installBtn) {
    installBtn.addEventListener('click', async () => {
      if (deferredPrompt) {
        const result = await deferredPrompt.prompt();
        deferredPrompt = null;
        installBtn.hidden = true;
      }
    });
  }
  
  window.addEventListener('appinstalled', () => {
    console.log('PWA was installed');
  });
  
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').catch(err => console.error('SW registration failed:', err));
  }
});
