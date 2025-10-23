// Version 2 - Enhanced Interface (Loading Screen Removed)

// Scroll Animations
class ScrollAnimations {
  constructor() {
    this.observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };
    this.init();
  }

  init() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, this.observerOptions);

    document.querySelectorAll('.fade-in').forEach(element => {
      observer.observe(element);
    });
  }
}

// Smooth Scrolling
class SmoothScrolling {
  constructor() {
    this.init();
  }

  init() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) {
          const headerHeight = document.querySelector('.nav').offsetHeight;
          const targetPosition = target.offsetTop - headerHeight - 20;
          
          window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
          });
        }
      });
    });
  }
}

// Website Stats
class WebsiteStats {
  constructor() {
    this.init();
  }

  init() {
    // Add current time
    this.updateTime();
    
    // Update time every minute
    setInterval(() => {
      this.updateTime();
    }, 60000);
  }

  updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit'
    });
    
    // You can add time display if needed
    // document.getElementById('currentTime').textContent = timeString;
  }
}

// Main Initialization
document.addEventListener('DOMContentLoaded', () => {
  // Initialize scroll animations
  const scrollAnimations = new ScrollAnimations();
  const smoothScrolling = new SmoothScrolling();
  const websiteStats = new WebsiteStats();
  
  // Add hover effects to stats container
  document.querySelectorAll('.stats-container').forEach(container => {
    container.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-5px)';
      this.style.boxShadow = '0 10px 25px rgba(77, 163, 255, 0.2)';
    });
    container.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
      this.style.boxShadow = 'none';
    });
  });

  // Add hover effects to list items within scrollable content
  document.querySelectorAll('.scrollable-content li').forEach(item => {
    item.addEventListener('mouseenter', function() {
      this.style.transform = 'translateX(5px)';
      this.style.transition = 'transform 0.3s ease';
    });
    item.addEventListener('mouseleave', function() {
      this.style.transform = 'translateX(0)';
    });
  });

  // Add hover effects to project info sections
  document.querySelectorAll('.info-section').forEach(section => {
    section.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-5px)';
      this.style.boxShadow = '0 10px 25px rgba(77, 163, 255, 0.2)';
    });
    section.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
      this.style.boxShadow = 'none';
    });
  });

  // Add hover effects to project links
  document.querySelectorAll('.project-link').forEach(link => {
    link.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-3px)';
      this.style.boxShadow = '0 10px 25px var(--accent-glow)';
    });
    link.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
      this.style.boxShadow = 'none';
    });
  });

  // Add pulse animation for stats
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }
  `;
  document.head.appendChild(style);

  // Active navigation highlighting
  const updateActiveNav = () => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav a');
    
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;
      if (window.pageYOffset >= sectionTop - 200) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  };

  window.addEventListener('scroll', updateActiveNav);
  updateActiveNav();

  // Typewriter effect for hero section
  const typeWriter = () => {
    const text = "PhD Candidate @ University of Virginia | AI • BCI • Robotics";
    const element = document.querySelector('.hero-subtitle');
    if (!element) return;
    
    element.textContent = '';
    let i = 0;
    
    const typeInterval = setInterval(() => {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
      } else {
        clearInterval(typeInterval);
      }
    }, 50);
  };

  // Start typewriter effect after a short delay
  setTimeout(typeWriter, 500);
}); 