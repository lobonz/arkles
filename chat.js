class OfficeRequirementsChat {
    constructor() {
        this.requirements = {};
        this.conversationHistory = [];
        this.isProcessing = false;
        this.conversationId = null;
        
        // Initialize UI elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.requirementsContent = document.getElementById('requirementsContent');
        this.progressFill = document.getElementById('progressFill');
        this.exportButton = document.getElementById('exportButton');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.appContainer = document.getElementById('appContainer');
        this.startButton = document.getElementById('startButton');
        this.devCompleteButton = document.getElementById('devCompleteButton');
        this.devExplorerButton = document.getElementById('devExplorerButton');
        this.devExplorerButtonBottom = document.getElementById('devExplorerButtonBottom');
        this.devOverlay = document.getElementById('devOverlay');
        this.providerSelect = document.getElementById('providerSelect');
        this.modelSelect = document.getElementById('modelSelect');
        this.devConvList = document.getElementById('devConvList');
        this.devDetailTitle = document.getElementById('devDetailTitle');
        this.devMessages = document.getElementById('devMessages');
        this.devReq = document.getElementById('devReq');
        this.devReqJson = document.getElementById('devReqJson');
        this.devCloseBtn = document.getElementById('devCloseBtn');
        this.devRefreshBtn = document.getElementById('devRefreshBtn');
        this.copyReqBtn = document.getElementById('copyReqBtn');
        this.deleteConvBtn = document.getElementById('deleteConvBtn');
        this.devToolbar = document.getElementById('devToolbar');
        this._selectedConversationId = null;
        
        this.initializeEventListeners();
        this.initializeRequirementsPanel();
        // Initial welcome screen is visible; chat UI is hidden until Start
        // Populate model list on load for the default provider
        this.refreshModelOptions().then(() => {
            // If a stored choice exists (e.g., query param or previous session), restore it
            const storedModel = sessionStorage.getItem('dev.model');
            const storedProvider = sessionStorage.getItem('dev.provider');
            if (storedProvider && this.providerSelect) this.providerSelect.value = storedProvider;
            if (this.modelSelect && storedModel) {
                const exists = Array.from(this.modelSelect.options).some(o => o.value === storedModel);
                if (exists) this.modelSelect.value = storedModel;
            }
        });
    }
    
    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', this.autoResizeTextarea);
        this.exportButton.addEventListener('click', () => this.exportRequirements());
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startNewRun());
        }
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', async () => {
                const previous = this.modelSelect ? this.modelSelect.value : '';
                await this.refreshModelOptions();
                // Try to preserve previous selection if it still exists in the new list
                if (this.modelSelect && previous) {
                    const exists = Array.from(this.modelSelect.options).some(o => o.value === previous);
                    if (exists) this.modelSelect.value = previous;
                }
            });
        }
        if (this.devCompleteButton) {
            this.devCompleteButton.addEventListener('click', () => this.forceComplete());
        }
        if (this.devExplorerButton) {
            this.devExplorerButton.addEventListener('click', () => this.openDevExplorer());
        }
        if (this.devExplorerButtonBottom) {
            this.devExplorerButtonBottom.addEventListener('click', () => this.openDevExplorer());
        }
        // Persist selections to session storage
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', () => {
                sessionStorage.setItem('dev.provider', this.providerSelect.value);
            });
        }
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => {
                sessionStorage.setItem('dev.model', this.modelSelect.value);
            });
        }
        if (this.devCloseBtn) {
            this.devCloseBtn.addEventListener('click', () => this.closeDevExplorer());
        }
        if (this.devRefreshBtn) {
            this.devRefreshBtn.addEventListener('click', () => this.loadConversations());
        }
        if (this.copyReqBtn) {
            this.copyReqBtn.addEventListener('click', () => this.copyRequirementsJSON());
        }
        if (this.deleteConvBtn) {
            this.deleteConvBtn.addEventListener('click', () => this.deleteSelectedConversation());
        }
    }
    
    initializeRequirementsPanel() {
        const sections = [
            { id: 'business', title: 'Business Profile', icon: 'fas fa-building' },
            { id: 'space', title: 'Space Requirements', icon: 'fas fa-ruler-combined' },
            { id: 'location', title: 'Location Criteria', icon: 'fas fa-map-marker-alt' },
            { id: 'financial', title: 'Financial Parameters', icon: 'fas fa-dollar-sign' },
            { id: 'technology', title: 'Technology & Infrastructure', icon: 'fas fa-wifi' },
            { id: 'amenities', title: 'Building & Amenities', icon: 'fas fa-heart' },
            { id: 'compliance', title: 'Compliance & Sustainability', icon: 'fas fa-shield-alt' },
            { id: 'flexibility', title: 'Flexibility & Options', icon: 'fas fa-exchange-alt' }
        ];
        
        sections.forEach(section => {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'requirement-section';
            sectionDiv.innerHTML = `
                <h3><i class="${section.icon}"></i> ${section.title}</h3>
                <div id="${section.id}-items" class="section-items"></div>
            `;
            this.requirementsContent.appendChild(sectionDiv);
        });
    }
    
    addWelcomeMessage() {
        this.addMessage(
            'assistant',
            "Hi! I'm here to help you find the perfect office space. Tell me about your business and what you're looking for in an office. For example, you could say something like:\n\n• \"We're a 15-person tech startup looking for our first office\"\n• \"Our accounting firm needs to relocate to a more professional area\"\n• \"We need flexible space for hybrid working with good meeting rooms\"\n\nWhat kind of space are you looking for?"
        );
    }

    async refreshModelOptions() {
        try {
            const provider = this.providerSelect ? this.providerSelect.value : 'openrouter';
            if (this.modelSelect) {
                this.modelSelect.disabled = true;
                this.modelSelect.innerHTML = '';
                const opt = document.createElement('option');
                opt.textContent = 'Loading…';
                opt.value = '';
                this.modelSelect.appendChild(opt);
            }
            // Load models
            const res = await fetch('http://localhost:5000/models');
            const data = await res.json();
            if (provider === 'ollama') {
                const items = (data.ollama_models || []).map(m => ({ value: m, label: m }));
                this.populateModelSelect(items);
            } else {
                const list = data.openrouter_models || [];
                const items = list.map(m => ({ value: m, label: m }));
                this.populateModelSelect(items);
            }
        } catch (e) {
            this.populateModelSelect([{ value: '', label: 'Unavailable' }]);
        }
    }

    populateModelSelect(items) {
        if (!this.modelSelect) return;
        this.modelSelect.innerHTML = '';
        items.forEach(({ value, label }) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            this.modelSelect.appendChild(opt);
        });
        this.modelSelect.disabled = items.length === 0 || !items[0].value;
    }
    
    autoResizeTextarea() {
        const textarea = document.getElementById('messageInput');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;

        this.addMessage('user', message);
        this.messageInput.value = '';
        this.autoResizeTextarea();
        
        this.isProcessing = true;
        this.sendButton.disabled = true;
        this.showTypingIndicator();
        
        try {
            const response = await this.sendToServer(message, {
                // Always include current provider/model to avoid stale values
                provider: this.providerSelect ? this.providerSelect.value : undefined,
                model: this.modelSelect ? this.modelSelect.value : undefined,
            });
            this.hideTypingIndicator();
            
            if (response.success) {
                this.addMessage('assistant', response.reply);
                this.updateRequirements(response.requirements);
                this.updateProgress();
                this.conversationHistory = response.conversation_history;
                this.conversationId = response.conversation_id || this.conversationId;
                if (response.is_complete) {
                    setTimeout(() => {
                        this.showWelcomeWithCompletion(response.reply || 'Completed. You can start again to find another space.');
                    }, 400);
                }
            } else {
                this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('assistant', 'Sorry, I had trouble connecting. Please check if the server is running.');
            console.error('Error:', error);
        }
        
        this.isProcessing = false;
        this.sendButton.disabled = false;
        this.messageInput.focus();
    }
    
    async sendToServer(message, options = {}) {
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_history: this.conversationHistory,
                current_requirements: this.requirements,
                conversation_id: this.conversationId,
                force_complete: options.forceComplete === true,
                completion_mode: options.completionMode,
                interview_mode: options.interviewMode,
                provider: options.provider !== undefined ? options.provider : (this.providerSelect ? this.providerSelect.value : undefined),
                model: options.model !== undefined ? options.model : (this.modelSelect ? this.modelSelect.value : undefined)
            })
        });
        
        return await response.json();
    }

    startNewRun() {
        // Reset state
        this.requirements = {};
        this.conversationHistory = [];
        this.conversationId = null;
        this.progressFill.style.width = '0%';
        this.exportButton.disabled = true;
        this.chatMessages.innerHTML = '';
        document.querySelectorAll('.section-items').forEach(section => { section.innerHTML = ''; });

        // Swap UI: show app, hide welcome
        if (this.welcomeScreen) this.welcomeScreen.classList.add('hidden');
        if (this.appContainer) this.appContainer.classList.remove('hidden');
        if (this.devToolbar) this.devToolbar.classList.add('hidden');

        // Seed first assistant message in chat
        this.addWelcomeMessage();
        this.messageInput.focus();

        // Do not refresh model options here; it can reset user selection.
        // Models are loaded on page load and when provider changes.
    }

    showWelcomeWithCompletion(messageText) {
        if (this.appContainer) this.appContainer.classList.add('hidden');
        if (this.welcomeScreen) this.welcomeScreen.classList.remove('hidden');
        if (this.devToolbar) this.devToolbar.classList.remove('hidden');
        const subtitle = document.getElementById('welcomeSubtitle');
        if (subtitle) {
            subtitle.textContent = messageText || 'All set. Start another run whenever you like.';
        }
    }

    // ---------------- DEV EXPLORER ----------------
    openDevExplorer() {
        if (!this.devOverlay) return;
        this.devOverlay.classList.add('visible');
        this.loadConversations();
    }

    closeDevExplorer() {
        if (!this.devOverlay) return;
        this.devOverlay.classList.remove('visible');
    }

    async loadConversations() {
        try {
            if (!this.devConvList) return;
            this.devConvList.innerHTML = '<div style="padding:10px;color:#666;">Loading…</div>';
            const res = await fetch('http://localhost:5000/dev/conversations');
            const data = await res.json();
            const items = data.items || [];
            this.renderConversationList(items);
        } catch (e) {
            if (this.devConvList) this.devConvList.innerHTML = '<div style="padding:10px;color:#b00020;">Failed to load conversations.</div>';
        }
    }

    renderConversationList(items) {
        if (!this.devConvList) return;
        if (!items.length) {
            this.devConvList.innerHTML = '<div style="padding:10px;color:#666;">No conversations yet.</div>';
            return;
        }
        this.devConvList.innerHTML = '';
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'dev-item';
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; gap:8px; align-items:center;">
                  <div style="font-weight:600; color:#333; max-width: 230px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${this.escapeHtml(item.title || '')}">${this.escapeHtml(item.title || '')}</div>
                  <div>
                    <span class="badge ${item.is_complete ? 'complete' : 'incomplete'}">${item.is_complete ? 'Complete' : 'In progress'}</span>
                  </div>
                </div>
                <div style="margin-top:6px; color:#555; font-size:12px; display:flex; gap:10px;">
                  <span><i class="fas fa-clock"></i> ${this.escapeHtml(item.created_at || '')}</span>
                  <span><i class="fas fa-message"></i> ${item.message_count || 0} msgs</span>
                  <span><i class="fas fa-list-check"></i> ${item.filled_fields_count || 0}/${item.filled_fields_total || 0} fields</span>
                </div>
            `;
            div.addEventListener('click', () => this.loadConversationDetails(item.id));
            this.devConvList.appendChild(div);
        });
    }

    async loadConversationDetails(conversationId) {
        try {
            if (this.devMessages) this.devMessages.innerHTML = '<div style="padding:10px;color:#666;">Loading…</div>';
            if (this.devReqJson) this.devReqJson.textContent = '{}';
            const res = await fetch(`http://localhost:5000/dev/conversations/${conversationId}`);
            const data = await res.json();
            this.renderConversationDetails(data);
            this._selectedConversationId = conversationId;
        } catch (e) {
            if (this.devMessages) this.devMessages.innerHTML = '<div style="padding:10px;color:#b00020;">Failed to load conversation.</div>';
        }
    }

    renderConversationDetails(data) {
        if (!data) return;
        if (this.devDetailTitle) {
            const status = data.is_complete ? 'Complete' : 'In progress';
            this.devDetailTitle.textContent = `${status} • ${data.provider || ''}/${data.model || ''} • ${data.created_at || ''}`;
        }
        if (this.devMessages) {
            this.devMessages.innerHTML = '';
            (data.messages || []).forEach((m, idx) => {
                const el = document.createElement('div');
                const altClass = (idx % 2) === 1 ? 'alt' : '';
                el.className = `dev-msg ${m.role} ${altClass}`;
                const icon = m.role === 'user' ? 'fa-user' : 'fa-robot';
                el.innerHTML = `
                    <div class="avatar"><i class="fas ${icon}"></i></div>
                    <div>
                      <div style="font-size:12px; color:#667eea; margin-bottom:4px;">
                        ${this.escapeHtml(m.role)} • <span style="color:#999;">${this.escapeHtml(m.created_at || '')}</span>
                      </div>
                      <div class="bubble" style="white-space: pre-wrap;">${this.escapeHtml(m.content || '')}</div>
                    </div>
                `;
                this.devMessages.appendChild(el);
            });
            // ensure scrollable area shows latest
            this.devMessages.scrollTop = this.devMessages.scrollHeight;
        }
        if (this.devReqJson) {
            try {
                this.devReqJson.textContent = JSON.stringify(data.requirements_latest || {}, null, 2);
            } catch {
                this.devReqJson.textContent = '{}';
            }
        }
        this._lastDevReqObj = data.requirements_latest || {};
    }

    async deleteSelectedConversation() {
        const id = this._selectedConversationId;
        if (!id) return;
        const sure = confirm('Delete this conversation and all messages/requirements?');
        if (!sure) return;
        try {
            const res = await fetch(`http://localhost:5000/dev/conversations/${id}`, { method: 'DELETE' });
            const data = await res.json();
            if (data && data.success) {
                // Clear detail view
                if (this.devDetailTitle) this.devDetailTitle.textContent = 'Select a conversation';
                if (this.devMessages) this.devMessages.innerHTML = '';
                if (this.devReqJson) this.devReqJson.textContent = '{}';
                this._selectedConversationId = null;
                // Refresh list
                this.loadConversations();
            } else {
                alert('Delete failed');
            }
        } catch (e) {
            alert('Delete failed');
        }
    }

    copyRequirementsJSON() {
        try {
            const text = JSON.stringify(this._lastDevReqObj || {}, null, 2);
            navigator.clipboard.writeText(text);
        } catch(e) {
            // no-op
        }
    }

    escapeHtml(text) {
        if (text == null) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    async forceComplete() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        this.sendButton.disabled = true;
        this.showTypingIndicator();
        try {
            const response = await this.sendToServer('[developer] complete now', { forceComplete: true });
            this.hideTypingIndicator();
            if (response.success) {
                this.addMessage('assistant', response.reply);
                this.updateRequirements(response.requirements || {});
                this.updateProgress();
                this.conversationHistory = response.conversation_history || [];
                this.conversationId = response.conversation_id || this.conversationId;
                setTimeout(() => {
                    this.showWelcomeWithCompletion(response.reply || 'Completed (dev). Start another run to test again.');
                }, 300);
            } else {
                this.addMessage('assistant', 'Completion failed.');
            }
        } catch (e) {
            this.hideTypingIndicator();
            this.addMessage('assistant', 'Could not complete. Is the server running?');
        }
        this.isProcessing = false;
        this.sendButton.disabled = false;
    }
    
    addMessage(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = content.replace(/\n/g, '<br>');
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    updateRequirements(newRequirements) {
        // Merge new requirements with existing ones
        this.requirements = { ...this.requirements, ...newRequirements };
        
        // Clear existing items
        document.querySelectorAll('.section-items').forEach(section => {
            section.innerHTML = '';
        });
        
        // Map requirements to sections
        const sectionMapping = {
            business: ['employees', 'business_type', 'has_existing_space', 'current_location', 'current_cost'],
            space: ['required_size', 'working_style', 'layout_preference', 'meeting_rooms', 'growth_plans'],
            location: ['preferred_areas', 'transport_access', 'brand_image', 'proximity_requirements'],
            financial: ['budget_range', 'lease_term', 'incentives_needed', 'fitout_budget'],
            technology: ['internet_requirements', 'power_requirements', 'data_infrastructure'],
            amenities: ['building_services', 'shared_amenities', 'security_needs', 'natural_light'],
            compliance: ['sustainability_importance', 'green_credentials', 'compliance_needs'],
            flexibility: ['flexibility_needs', 'timeline', 'short_term_options']
        };
        
        // Populate requirements in their respective sections
        Object.entries(this.requirements).forEach(([key, value]) => {
            if (value && value.trim && value.trim() !== '') {
                const section = this.findSection(key, sectionMapping);
                if (section) {
                    this.addRequirementItem(section, key, value);
                }
            }
        });
    }
    
    findSection(key, mapping) {
        for (const [section, keys] of Object.entries(mapping)) {
            if (keys.includes(key)) {
                return section;
            }
        }
        return 'business'; // Default section
    }
    
    addRequirementItem(sectionId, key, value) {
        const sectionItems = document.getElementById(`${sectionId}-items`);
        if (!sectionItems) return;
        
        const item = document.createElement('div');
        item.className = 'requirement-item filled';
        
        const label = this.formatLabel(key);
        item.innerHTML = `
            <span class="requirement-label">${label}</span>
            <span class="requirement-value">${value}</span>
        `;
        
        sectionItems.appendChild(item);
    }
    
    formatLabel(key) {
        const labels = {
            employees: 'Staff Count',
            business_type: 'Business Type',
            has_existing_space: 'Current Space',
            current_location: 'Current Location',
            current_cost: 'Current Cost',
            required_size: 'Required Size',
            working_style: 'Working Style',
            layout_preference: 'Layout',
            meeting_rooms: 'Meeting Rooms',
            growth_plans: 'Growth Plans',
            preferred_areas: 'Preferred Areas',
            transport_access: 'Transport',
            brand_image: 'Brand Image',
            proximity_requirements: 'Proximity Needs',
            budget_range: 'Budget',
            lease_term: 'Lease Term',
            incentives_needed: 'Incentives',
            fitout_budget: 'Fit-out Budget',
            internet_requirements: 'Internet',
            power_requirements: 'Power',
            data_infrastructure: 'Data Infrastructure',
            building_services: 'Building Services',
            shared_amenities: 'Amenities',
            security_needs: 'Security',
            natural_light: 'Natural Light',
            sustainability_importance: 'Sustainability',
            green_credentials: 'Green Features',
            compliance_needs: 'Compliance',
            flexibility_needs: 'Flexibility',
            timeline: 'Timeline',
            short_term_options: 'Short-term Options'
        };
        
        return labels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    updateProgress() {
        const totalPossibleRequirements = 30; // Approximate total requirements
        const filledRequirements = Object.keys(this.requirements).filter(key => 
            this.requirements[key] && this.requirements[key].toString().trim() !== ''
        ).length;
        
        const progress = Math.min((filledRequirements / totalPossibleRequirements) * 100, 100);
        this.progressFill.style.width = `${progress}%`;
        
        // Enable export when we have substantial information
        this.exportButton.disabled = filledRequirements < 5;
    }
    
    exportRequirements() {
        let report = '# Office Premises Requirements Report\n\n';
        report += `**Generated on:** ${new Date().toLocaleDateString()}\n\n`;
        
        const sections = [
            { title: 'Business Profile', keys: ['employees', 'business_type', 'has_existing_space', 'current_location'] },
            { title: 'Space Requirements', keys: ['required_size', 'working_style', 'layout_preference', 'meeting_rooms'] },
            { title: 'Location Criteria', keys: ['preferred_areas', 'transport_access', 'brand_image'] },
            { title: 'Financial Parameters', keys: ['budget_range', 'lease_term', 'incentives_needed'] },
            { title: 'Technology & Infrastructure', keys: ['internet_requirements', 'power_requirements'] },
            { title: 'Building & Amenities', keys: ['building_services', 'shared_amenities', 'security_needs'] },
            { title: 'Compliance & Sustainability', keys: ['sustainability_importance', 'green_credentials'] },
            { title: 'Flexibility & Options', keys: ['flexibility_needs', 'timeline'] }
        ];
        
        sections.forEach(section => {
            const sectionData = section.keys.filter(key => this.requirements[key]);
            if (sectionData.length > 0) {
                report += `## ${section.title}\n\n`;
                sectionData.forEach(key => {
                    const label = this.formatLabel(key);
                    const value = this.requirements[key];
                    report += `**${label}:** ${value}\n\n`;
                });
            }
        });
        
        // Add conversation summary
        report += '## Conversation Summary\n\n';
        this.conversationHistory.forEach((msg, index) => {
            if (index % 2 === 0) { // User messages
                report += `**Customer:** ${msg.content}\n\n`;
            }
        });
        
        // Download the report
        const blob = new Blob([report], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `office-requirements-${new Date().toISOString().split('T')[0]}.md`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize the chat application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new OfficeRequirementsChat();
});
