document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const button = document.getElementById('submit-button');
  if (!form || !button) return;
  form.addEventListener('submit', () => {
    if (!form.reportValidity()) return;
    button.disabled = true;
    button.querySelector('span:first-child').textContent = 'Both models are predicting…';
    button.insertAdjacentHTML(
      'beforebegin',
      '<p class="loading-note" role="status">Running TabPFN and TabFM concurrently. CPU predictions can take several minutes.</p>',
    );
  });
});
