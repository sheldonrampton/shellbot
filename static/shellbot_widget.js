document.addEventListener("DOMContentLoaded", () => {
  const chatBtn = document.createElement('button');
  chatBtn.textContent = "💬 Chat with Shellbot";
  chatBtn.style.cssText = `
    position:fixed;bottom:20px;right:20px;padding:12px 18px;
    background:#222;color:white;border:none;border-radius:25px;
    font-size:16px;cursor:pointer;z-index:9999;
    box-shadow:0 4px 8px rgba(0,0,0,0.3);
  `;
  document.body.appendChild(chatBtn);

  const iframe = document.createElement('iframe');
  iframe.src = 'https://www.shellbot.ai';
  iframe.style.cssText = `
    position:fixed;bottom:80px;right:20px;width:400px;height:600px;
    border:1px solid #444;border-radius:10px;display:none;
    z-index:9998;box-shadow:0 4px 12px rgba(0,0,0,0.4);
  `;
  document.body.appendChild(iframe);

  chatBtn.onclick = () => {
    iframe.style.display = (iframe.style.display === 'none') ? 'block' : 'none';
  };
});
