import { NavLink } from "react-router-dom";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { isDesktop, isTablet } from "react-device-detect";
import { TooltipPortal } from "@radix-ui/react-tooltip";
import { NavData } from "@/types/navigation";
import { IconType } from "react-icons";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";

const variants = {
  primary: {
    active: "font-bold text-white bg-selected hover:bg-selected/80",
    inactive: "text-secondary-foreground bg-secondary hover:bg-muted",
  },
  secondary: {
    active: "font-bold text-selected",
    inactive: "text-secondary-foreground",
  },
};

type NavItemProps = {
  className?: string;
  item: NavData;
  Icon: IconType;
  onClick?: () => void;
};

export default function NavItem({
  className,
  item,
  Icon,
  onClick,
}: NavItemProps) {
  const { t } = useTranslation(["common"]);
  if (item.enabled == false) {
    return;
  }

  const label = t(item.title);
  const content = (
    <NavLink
      to={item.url}
      onClick={onClick}
      aria-label={label}
      title={isDesktop ? undefined : label}
      className={({ isActive }) =>
        cn(
          "flex flex-col items-center justify-center rounded-lg p-[6px]",
          className,
          variants[item.variant ?? "primary"][isActive ? "active" : "inactive"],
        )
      }
    >
      <Icon className="size-5" />
      {item.url === "/recap" && item.variant === "secondary" && (
        <span className="mt-0.5 hidden text-[10px] leading-none md:block">
          {label}
        </span>
      )}
    </NavLink>
  );

  if (isDesktop || isTablet) {
    return (
      <Tooltip>
        <TooltipTrigger>{content}</TooltipTrigger>
        <TooltipPortal>
          <TooltipContent side={item.variant === "secondary" ? "top" : "right"}>
            <p>{t(item.title)}</p>
          </TooltipContent>
        </TooltipPortal>
      </Tooltip>
    );
  }

  return content;
}
