<template>
  <div ref="root" class="custom-select" :class="{ 'is-disabled': disabled }">
    <button
      type="button"
      class="select-trigger"
      :disabled="disabled"
      @click="toggle"
    >
      <span class="select-text">
        <span class="select-label">{{ selectedOption?.label || placeholder }}</span>
        <span v-if="selectedOption?.detail" class="select-detail">{{ selectedOption.detail }}</span>
      </span>
      <svg class="select-chevron" viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6 9 6 6 6-6" />
      </svg>
    </button>

    <Transition name="select-pop">
      <div v-if="open" class="select-menu">
        <button
          v-for="option in options"
          :key="String(option.value)"
          type="button"
          class="select-option"
          :class="{ 'is-selected': option.value === modelValue }"
          @click="choose(option.value)"
        >
          <span>
            <span class="option-label">{{ option.label }}</span>
            <span v-if="option.detail" class="option-detail">{{ option.detail }}</span>
          </span>
          <svg v-if="option.value === modelValue" class="check-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M20 6 9 17l-5-5" />
          </svg>
        </button>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  modelValue: { type: [String, Number], default: '' },
  options: { type: Array, default: () => [] },
  placeholder: { type: String, default: '请选择' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue'])
const root = ref(null)
const open = ref(false)

const selectedOption = computed(() =>
  props.options.find(option => option.value === props.modelValue)
)

function toggle() {
  if (props.disabled) return
  open.value = !open.value
}

function choose(value) {
  emit('update:modelValue', value)
  open.value = false
}

function onPointerDown(event) {
  if (!root.value || root.value.contains(event.target)) return
  open.value = false
}

onMounted(() => document.addEventListener('pointerdown', onPointerDown))
onBeforeUnmount(() => document.removeEventListener('pointerdown', onPointerDown))
</script>

<style scoped>
.custom-select {
  position: relative;
  width: 100%;
  min-width: 0;
}

.select-trigger {
  display: flex;
  width: 100%;
  height: 42px;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  background: #fff;
  color: #0f172a;
  padding: 0 12px 0 14px;
  font: inherit;
  cursor: pointer;
  transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}

.select-trigger:hover:not(:disabled) {
  border-color: #94a3b8;
  background: #f8fafc;
}

.select-trigger:focus-visible {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, .16);
}

.select-text {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  align-items: flex-start;
}

.select-label {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
  font-weight: 600;
}

.select-detail {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-top: 1px;
  color: #64748b;
  font-size: 11px;
}

.select-chevron,
.check-icon {
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 2;
}

.select-chevron {
  color: #64748b;
}

.select-menu {
  position: absolute;
  z-index: 30;
  top: calc(100% + 8px);
  left: 0;
  width: 100%;
  max-height: 260px;
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  padding: 6px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, .12);
}

.select-option {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #0f172a;
  padding: 9px 10px;
  text-align: left;
  cursor: pointer;
}

.select-option > span {
  min-width: 0;
}

.select-option:hover {
  background: #f8fafc;
}

.select-option.is-selected {
  background: #eef2ff;
  color: #3730a3;
}

.option-label {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
  font-weight: 600;
}

.option-detail {
  display: block;
  margin-top: 2px;
  color: #64748b;
  font-size: 12px;
}

.is-disabled {
  opacity: .65;
}

.select-pop-enter-active,
.select-pop-leave-active {
  transition: opacity .14s ease, transform .14s ease;
}

.select-pop-enter-from,
.select-pop-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
