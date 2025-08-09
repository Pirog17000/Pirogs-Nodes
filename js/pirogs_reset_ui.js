// Pirog's Nodes Custom UI with Reset Buttons
// Based on MX Toolkit approach for custom ComfyUI visual controls
import { app } from "../../../scripts/app.js";

class PirogsResetUI {
    constructor(node) {
        this.node = node;
        this.resetButtons = new Map(); // Track reset buttons for each input
        this.originalDefaults = new Map(); // Store original default values
        
        // Set up gutter width first
        this._reset_gutter_width = 30; // px reserved on the right
        this._overlay_shrink = 18; // visually hide a bit of the native widget before the gutter
        
        // Capture defaults from current widget values at creation time
        // (Front-end does not have access to Python-side INPUT_TYPES here)
        this.captureWidgetDefaults();
        
        // Reorder widgets to put seed and strength at bottom
        this.reorderWidgets();
        
        // Visually shrink widgets so they don't overlap the reset gutter
        this.shrinkWidgetWidths();
        
        // Also try delayed shrinking after node is fully loaded
        setTimeout(() => {
            this.shrinkWidgetWidths();
            this.adjustNodeWidgetArea();
            this.forceWidgetResize();
        }, 100);
        
        // Override the node's drawing and interaction methods
        this.setupCustomUI();
    }
    
    captureWidgetDefaults() {
        if (!this.node.widgets) return;
        for (const widget of this.node.widgets) {
            if (widget?.type === "number" && widget?.name) {
                // Prefer an explicit default on the widget options if present
                const explicitDefault = widget.options?.default;
                const initialValue = widget.value;
                const defaultValue = (explicitDefault !== undefined) ? explicitDefault : initialValue;
                this.originalDefaults.set(widget.name, defaultValue);
            }
        }
    }
    
    shrinkWidgetWidths() {
        if (!this.node.widgets) return;
        const gutter = this._reset_gutter_width || 30;
        const shrinkBy = gutter + 4; // a little extra padding before gutter
        
        for (const widget of this.node.widgets) {
            if (!widget || widget._pirog_shrunk) continue;
            
            // ONLY modify number widgets - leave combo boxes and other widgets alone
            if (widget.type !== "number") continue;
            
            // Directly modify widget properties if they exist
            if (widget.options) {
                widget.options.max_width = (widget.options.max_width || 200) - shrinkBy;
            }
            
            // Set a custom property to track the reduced width
            widget._pirog_reduced_width = shrinkBy;
            
            // Override the draw method to respect the reduced width
            if (typeof widget.draw === "function") {
                const original = widget.draw.bind(widget);
                widget.draw = function(ctx, node, x, y, w, h) {
                    // Use forced width if available, otherwise calculate
                    const newW = widget._forced_width || Math.max(50, (w || node.size[0]) - shrinkBy);
                    return original(ctx, node, x, y, newW, h);
                };
            }
            
            // Also try overriding the mouse method to prevent interaction in the gutter area
            if (typeof widget.mouse === "function") {
                const originalMouse = widget.mouse.bind(widget);
                widget.mouse = function(event, pos, node) {
                    // Check if mouse is in the gutter area
                    const gutterStart = node.size[0] - gutter;
                    if (pos[0] > gutterStart) {
                        return false; // Don't handle mouse events in gutter
                    }
                    return originalMouse(event, pos, node);
                };
            }
            
            widget._pirog_shrunk = true;
        }
    }
    
    adjustNodeWidgetArea() {
        // Try to modify node's internal widget area properties
        if (this.node.widget_area) {
            this.node.widget_area.width = Math.max(50, this.node.widget_area.width - 34);
        }
        
        // Force node to recalculate widget positions
        if (this.node.setDirtyCanvas) {
            this.node.setDirtyCanvas(true);
        }
        
        // Try to modify the node's effective size for widgets
        if (this.node.widgets) {
            this.node._widget_width = Math.max(50, (this.node.size[0] || 200) - 34);
        }
    }
    
    forceWidgetResize() {
        // Directly manipulate widget dimensions as a last resort
        if (!this.node.widgets) return;
        
        for (const widget of this.node.widgets) {
            if (widget.type === "number") {
                // Instead of setting fixed width, override the width calculation dynamically
                this.makeWidgetResponsive(widget);
            }
        }
        
        // Force the node to recalculate layout
        if (this.node.setDirtyCanvas) {
            this.node.setDirtyCanvas(true, true);
        }
    }
    
    makeWidgetResponsive(widget) {
        // Override the widget's width calculation to be dynamic
        Object.defineProperty(widget, '_forced_width', {
            get: () => {
                // Always calculate based on current node size
                return Math.max(50, (this.node.size[0] || 200) - 30);
            }
        });
        
        // Also override the width property to be dynamic
        Object.defineProperty(widget, 'width', {
            get: () => {
                return Math.max(50, (this.node.size[0] || 200) - 30);
            },
            set: () => {
                // Ignore attempts to set fixed width
            }
        });
        
        // Mark that this widget has been made responsive
        widget._pirog_responsive = true;
    }
    
    updateWidgetWidths() {
        // Force redraw to apply new width calculations
        if (this.node.setDirtyCanvas) {
            this.node.setDirtyCanvas(true, true);
        }
        
        // Also force widgets to recalculate if they have the method
        if (this.node.widgets) {
            for (const widget of this.node.widgets) {
                if (widget._pirog_responsive && widget.setDirtyCanvas) {
                    widget.setDirtyCanvas(true);
                }
            }
        }
    }
    
    reorderWidgets() {
        if (!this.node.widgets) return;
        
        // Find seed and strength widgets
        const seedWidget = this.node.widgets.find(w => w.name === "seed");
        const strengthWidget = this.node.widgets.find(w => w.name === "strength");
        
        if (seedWidget || strengthWidget) {
            // Remove seed and strength from current positions
            this.node.widgets = this.node.widgets.filter(w => w.name !== "seed" && w.name !== "strength");
            
            // Add strength first, then seed at the end
            if (strengthWidget) {
                this.node.widgets.push(strengthWidget);
            }
            if (seedWidget) {
                this.node.widgets.push(seedWidget);
            }
        }
    }
    
    setupCustomUI() {
        const originalOnDrawForeground = this.node.onDrawForeground;
        const originalOnMouseDown = this.node.onMouseDown;
        const originalComputeSize = this.node.computeSize;
        
        // Override computeSize to account for reset buttons
        this.node.computeSize = (minHeight) => {
            const size = originalComputeSize ? originalComputeSize.call(this.node, minHeight) : [LiteGraph.NODE_WIDTH, LiteGraph.NODE_HEIGHT];
            // Ensure there is enough total width to include the gutter, but don't keep growing
            size[0] = Math.max(size[0], LiteGraph.NODE_WIDTH + this._reset_gutter_width);
            return size;
        };
        
        // CRITICAL: Override widget sizing by intercepting the widget creation and layout
        // Hook into widget size calculations when they're being laid out
        if (this.node.addWidget) {
            const originalAddWidget = this.node.addWidget.bind(this.node);
            this.node.addWidget = function(type, name, value, callback, options) {
                const widget = originalAddWidget(type, name, value, callback, options);
                
                // For number widgets, override their size calculation
                if (widget && widget.type === "number") {
                    const originalComputeSize = widget.computeSize;
                    widget.computeSize = function(ctx) {
                        const originalSize = originalComputeSize ? originalComputeSize.call(this, ctx) : [200, LiteGraph.NODE_WIDGET_HEIGHT];
                        // Reduce width by reset button space
                        originalSize[0] = Math.max(50, originalSize[0] - 30);
                        return originalSize;
                    };
                }
                
                return widget;
            };
        }
        
        // Note: Removed global widget clipping - we handle width on a per-widget basis now
        
        // Custom drawing for reset buttons
        this.node.onDrawForeground = (ctx) => {
            if (originalOnDrawForeground) {
                originalOnDrawForeground.call(this.node, ctx);
            }
            
            if (this.node.flags.collapsed) return;
            
            this.drawResetButtons(ctx);
        };
        
        // Handle mouse clicks for reset buttons - only intercept reset button areas specifically
        this.node.onMouseDown = (e) => {
            // First check if it's specifically on a reset button
            const resetClicked = this.handleResetButtonClick(e);
            if (resetClicked) return true;
            
            // Allow all other clicks to go through normally (including increment arrows)
            if (originalOnMouseDown) {
                return originalOnMouseDown.call(this.node, e);
            }
            return false;
        };
        
        // Hook into resize events to update widget widths dynamically
        const originalOnResize = this.node.onResize;
        this.node.onResize = (size) => {
            if (originalOnResize) {
                originalOnResize.call(this.node, size);
            }
            
            // Force widget layout recalculation when node is resized
            this.updateWidgetWidths();
        };
        
        // Also hook into size changes
        const originalSetSize = this.node.setSize;
        if (originalSetSize) {
            this.node.setSize = (size) => {
                originalSetSize.call(this.node, size);
                this.updateWidgetWidths();
            };
        }
    }
    
    drawResetButtons(ctx) {
        if (!this.node.widgets) return;
        
        const buttonSize = 14; // round button
        const rightMargin = 6; // inner padding from the gutter edge
        const widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
        const gutterWidth = this._reset_gutter_width;
        const gutterX = this.node.size[0] - gutterWidth; // left side of gutter

        for (let i = 0; i < this.node.widgets.length; i++) {
            const widget = this.node.widgets[i];
            
            // Skip hidden widgets and certain types
            if (widget.type === "hidden" || widget.hidden) {
                continue;
            }
            // Only add reset buttons for numeric inputs (INT, FLOAT)
            // Debug: Log widget types to see what's happening
            console.log(`Widget: ${widget.name}, Type: ${widget.type}, HasDefault: ${this.originalDefaults.has(widget.name)}`);
            
            if (widget.type === "number" && this.originalDefaults.has(widget.name)) {
                // Align to the actual widget row using last_y set by Comfy draw cycle
                const rowY = (widget.last_y ?? (this.node.widgets_start_y || widgetHeight));

                // Paint a small gutter strip just for the reset button (don't hide increment arrow)
                const resetButtonArea = 20; // Just enough for reset button
                const resetGutterX = this.node.size[0] - resetButtonArea;
                ctx.fillStyle = this.node.bgcolor || "#2a2a2a";
                ctx.fillRect(resetGutterX, rowY + 1, resetButtonArea, widgetHeight - 2);
                const centerX = this.node.size[0] - (resetButtonArea / 2); // Use the smaller reset button area
                const centerY = rowY + widgetHeight / 2;
                const buttonX = centerX - buttonSize / 2;
                const buttonY = centerY - buttonSize / 2;

                // Store button position for click detection
                this.resetButtons.set(widget.name, {
                    x: buttonX,
                    y: buttonY,
                    width: buttonSize,
                    height: buttonSize,
                    widget: widget
                });

                // Check if current value differs from default
                const defaultValue = this.originalDefaults.get(widget.name);
                const isDifferent = !(Number.isFinite(widget.value) && Math.abs(widget.value - defaultValue) <= 0.0001);

                // Draw circular button
                ctx.fillStyle = isDifferent ? "rgba(255, 140, 0, 0.95)" : "rgba(128, 128, 128, 0.65)";
                ctx.beginPath();
                ctx.arc(centerX, centerY, buttonSize / 2, 0, 2 * Math.PI);
                ctx.fill();

                // Border
                ctx.strokeStyle = isDifferent ? "rgba(255, 160, 40, 1)" : "rgba(160, 160, 160, 0.8)";
                ctx.lineWidth = 1;
                ctx.stroke();

                // Reset arrow symbol ↺
                ctx.fillStyle = isDifferent ? "#fff" : "rgba(220, 220, 220, 0.95)";
                ctx.font = "12px Arial, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("↺", centerX, centerY);
            } else {
                // Clear any stale button rect for non-number widgets
                this.resetButtons.delete(widget.name);
            }
        }
    }
    
    handleResetButtonClick(e) {
        const clickX = e.canvasX - this.node.pos[0];
        const clickY = e.canvasY - this.node.pos[1];
        const gutterWidth = this._reset_gutter_width;
        const gutterX = this.node.size[0] - gutterWidth;
        const overlayX = gutterX - this._overlay_shrink;
        
        // If the click is inside the overlay or gutter area, treat it as a reset click for the row
        if (clickX >= overlayX) {
            // Find the button whose row contains the click's Y
            for (const [widgetName, buttonInfo] of this.resetButtons.entries()) {
                const withinRow = clickY >= (buttonInfo.y - 4) && clickY <= (buttonInfo.y + buttonInfo.height + 4);
                if (withinRow) {
                    const defaultValue = this.originalDefaults.get(widgetName);
                    if (defaultValue !== undefined) {
                        const safeValue = Number.isFinite(defaultValue) ? defaultValue : 0;
                        buttonInfo.widget.value = safeValue;

                        if (buttonInfo.widget.callback) {
                            buttonInfo.widget.callback(safeValue, app.canvas, this.node, this.node.pos, e);
                        }

                        app.graph.setDirtyCanvas(true, true);
                        this.node.setDirtyCanvas(true, true);
                    }
                    return true;
                }
            }
            // Consume the click even if no button matched to avoid hitting the slider increment area
            return true;
        }

        for (const [widgetName, buttonInfo] of this.resetButtons.entries()) {
            if (clickX >= buttonInfo.x && clickX <= buttonInfo.x + buttonInfo.width &&
                clickY >= buttonInfo.y && clickY <= buttonInfo.y + buttonInfo.height) {
                
                // Reset the widget to its default value
                const defaultValue = this.originalDefaults.get(widgetName);
                if (defaultValue !== undefined) {
                    // Some widgets can hold NaN if user cleared text; normalize
                    const safeValue = Number.isFinite(defaultValue) ? defaultValue : 0;
                    buttonInfo.widget.value = safeValue;
                    
                    // Trigger change callback to update the node
                    if (buttonInfo.widget.callback) {
                        buttonInfo.widget.callback(safeValue, app.canvas, this.node, this.node.pos, e);
                    }
                    
                    // Mark graph as changed
                    app.graph.setDirtyCanvas(true, true);
                    
                    // Force redraw
                    this.node.setDirtyCanvas(true, true);
                }
                
                return true; // Consumed the click
            }
        }
        
        return false; // Click was not on a reset button
    }
}

// Register the extension for all Pirog nodes
app.registerExtension({
    name: "pirog.reset_ui",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to all nodes in the pirog category
        if (nodeData.category?.startsWith?.("pirog/")) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Initialize reset UI for this node
                this.pirogsResetUI = new PirogsResetUI(this);
            };
        }
    }
});