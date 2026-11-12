
const links = [...document.querySelectorAll('.rail nav a')];
const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
const spy = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const i = sections.indexOf(entry.target);
    if (i < 0) return;
    links.forEach(l => l.classList.remove('on'));
    links[i].classList.add('on');
  });
}, { rootMargin: '-12% 0px -70% 0px', threshold: 0 });
sections.forEach(s => spy.observe(s));
