import{r as i,j as e}from"./index-BJHAE5s4.js";import{l as y,n as x,o as f,p as S,_ as w,q as a,O as g,M as j,L as k,S as M}from"./components-XrQPQ4fj.js";/**
 * @remix-run/react v2.17.4
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */let l="positions";function L({getKey:t,...c}){let{isSpaMode:u}=y(),o=x(),m=f();S({getKey:t,storageKey:l});let d=i.useMemo(()=>{if(!t)return null;let s=t(o,m);return s!==o.key?s:null},[]);if(u)return null;let h=((s,p)=>{if(!window.history.state||!window.history.state.key){let r=Math.random().toString(32).slice(2);window.history.replaceState({key:r},"")}try{let n=JSON.parse(sessionStorage.getItem(s)||"{}")[p||window.history.state.key];typeof n=="number"&&window.scrollTo(0,n)}catch(r){console.error(r),sessionStorage.removeItem(s)}}).toString();return i.createElement("script",w({},c,{suppressHydrationWarning:!0,dangerouslySetInnerHTML:{__html:`(${h})(${a(JSON.stringify(l))}, ${a(JSON.stringify(d))})`}}))}const R=()=>[{rel:"stylesheet",href:"/tailwind.css"}];function _({children:t}){return e.jsxs("html",{lang:"en",className:"h-full",children:[e.jsxs("head",{children:[e.jsx("meta",{charSet:"utf-8"}),e.jsx("meta",{name:"viewport",content:"width=device-width, initial-scale=1"}),e.jsx(j,{}),e.jsx(k,{})]}),e.jsxs("body",{className:"h-full bg-gray-950 text-gray-100",children:[t,e.jsx(L,{}),e.jsx(M,{})]})]})}function b(){return e.jsx(g,{})}export{_ as Layout,b as default,R as links};
