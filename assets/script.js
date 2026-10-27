document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const button = document.getElementById('submit-button');
  if (!form || !button) return;
  form.addEventListener('submit', () => {
    if (!form.reportValidity()) return;
    button.disabled = true;
    button.querySelector('span:first-child').textContent = 'Models are predicting…';
    button.insertAdjacentHTML('beforebegin', '<p class="loading-note" role="status">Running actual model inference. First requests may take a little longer.</p>');
  });
});
