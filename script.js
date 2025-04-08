
window.addEventListener("load", () => {
  document.getElementById("preloader").style.display = "none";
});

const toggle = document.getElementById("modeToggle");
toggle.addEventListener("change", () => {
  document.body.classList.toggle("dark-mode");
});

const revealElements = document.querySelectorAll(".reveal");
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add("active");
    }
  });
}, { threshold: 0.1 });

revealElements.forEach(el => observer.observe(el));
